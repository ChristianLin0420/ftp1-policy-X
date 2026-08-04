"""Precompute a frozen embedding for every instruction string in the corpus.

Stage 3 aligns a clip representation to its instruction. The vocabulary is small -- 3,104
unique strings across 25,960 episodes -- so the embeddings are computed **once, offline**
and looked up as a frozen table. No text model ever runs inside the training loop.

The default encoder is SigLIP's text tower rather than a causal LM. That is a deliberate
change from the original plan, for a reason that showed up on inspection: Stage 3's objective
is cosine retrieval, and mean-pooled causal-LM hidden states are strongly anisotropic -- they
crowd into a narrow cone where cosine similarity separates poorly. SigLIP's text tower is
contrastively pretrained, so its space is already organized the way the objective reads it.

The competing argument was to match FTP-1's own language space (Gemma) so a later policy
graft inherits it. That argument is not load-bearing yet, because Stage 4 was scoped to the
world model rather than the FTP-1 graft; ``--model`` exists for when it becomes so.

Usage::

    uv run python scripts/mot_jepa_embed_instructions.py --clips /lustre/.../ftp1-clips
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
import zarr

DEFAULT_MODEL = "google/siglip-base-patch16-384"
#: SigLIP was trained with fixed-length padding to 64; matching that at inference matters,
#: because its pooled output is sensitive to the padding regime it saw during training.
MAX_TOKENS = 64


def embed_texts(texts: list[str], model_name: str, *, batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    text_model = getattr(model, "text_model", model).to(device).eval()

    out: list[np.ndarray] = []
    for begin in range(0, len(texts), batch_size):
        batch = texts[begin : begin + batch_size]
        tokens = tokenizer(batch, padding="max_length", max_length=MAX_TOKENS, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            hidden = text_model(**tokens)
        pooled = getattr(hidden, "pooler_output", None)
        if pooled is None:  # a causal LM has no pooler; mean-pool over real tokens
            mask = tokens["attention_mask"][..., None].to(hidden.last_hidden_state.dtype)
            pooled = (hidden.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
        print(f"    {min(begin + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return np.concatenate(out, axis=0)


def report_paraphrase_geometry(embeddings: np.ndarray, store_ids: dict[str, list[int]]) -> None:
    """Sanity-check that paraphrases of one task sit closer together than across tasks.

    This is the check that decides whether the embedding space is usable at all. If a store's
    own paraphrases are no closer to each other than to another store's, then Stage 3's
    positives are not actually positives and the retrieval number would be meaningless.
    """
    unit = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(1e-9)
    rng = np.random.default_rng(0)
    usable = {name: ids for name, ids in store_ids.items() if len(ids) >= 2}
    if len(usable) < 2:
        print("  (too few multi-paraphrase stores to compare)")
        return

    within, across = [], []
    names = sorted(usable)
    for name in names:
        ids = usable[name]
        pick = rng.choice(ids, size=min(len(ids), 20), replace=False)
        block = unit[pick]
        similarity = block @ block.T
        upper = similarity[np.triu_indices(len(pick), k=1)]
        within.append(upper.mean())

        other = usable[names[(names.index(name) + 1) % len(names)]]
        across.append((block @ unit[rng.choice(other, size=min(len(other), 20), replace=False)].T).mean())

    within_mean, across_mean = float(np.mean(within)), float(np.mean(across))
    print(f"  mean cos within a store's paraphrases : {within_mean:.4f}")
    print(f"  mean cos across different stores      : {across_mean:.4f}")
    print(f"  separation                            : {within_mean - across_mean:+.4f}")
    if within_mean <= across_mean:
        print("  WARNING: paraphrases are not closer than unrelated tasks; Stage 3 positives are not positives")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Defaults to <clips>/instruction_emb.npz")
    args = parser.parse_args()

    vocabulary_path = args.clips / "instructions.json"
    if not vocabulary_path.exists():
        parser.error(f"{vocabulary_path} missing; run scripts/mot_jepa_add_conditioning.py first")
    texts = json.loads(vocabulary_path.read_text())["instructions"]
    print(f"embedding {len(texts)} instructions with {args.model} on {args.device}")

    embeddings = embed_texts(texts, args.model, batch_size=args.batch_size, device=args.device)
    if embeddings.shape[0] != len(texts):
        raise RuntimeError(f"got {embeddings.shape[0]} embeddings for {len(texts)} texts")

    # Which instruction ids belong to which store -- needed for the geometry check and, in
    # Stage 3, for labelling clips by task rather than by string.
    store_ids: dict[str, list[int]] = {}
    for store in sorted(args.clips.glob("*/*.zarr")):
        try:
            ids = np.asarray(zarr.open(str(store), mode="r")["meta/instruction_id"][:])
        except Exception as exc:
            print(f"  [warn] {store.name}: {type(exc).__name__}: {exc}")
            continue
        store_ids[f"{store.parent.name}/{store.name}"] = sorted({int(i) for i in ids})

    print(f"\nparaphrase geometry over {len(store_ids)} stores:")
    report_paraphrase_geometry(embeddings, store_ids)

    output = args.output or (args.clips / "instruction_emb.npz")
    np.savez(
        output,
        embeddings=embeddings.astype(np.float32),
        model=np.asarray(args.model),
        store_names=np.asarray(sorted(store_ids), dtype=object),
        store_instruction_ids=np.asarray([store_ids[name] for name in sorted(store_ids)], dtype=object),
    )
    print(f"\nwrote {output}  shape={embeddings.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
