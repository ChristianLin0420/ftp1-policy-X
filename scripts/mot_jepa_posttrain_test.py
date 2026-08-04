from __future__ import annotations

import types

import numpy as np
import pytest

from scripts.mot_jepa_posttrain import fixed_eval_indices
from scripts.mot_jepa_posttrain import split_tasks


def fake_dataset(entries: np.ndarray):
    """Just enough of MotJepaClipDataset for the index selection: a clip_index."""
    return types.SimpleNamespace(clip_index=types.SimpleNamespace(entries=entries))


# --------------------------------------------------------------------------------------
# The train/eval split
# --------------------------------------------------------------------------------------


def test_split_is_by_task_not_by_store():
    """Two stores of one task must land on the same side.

    Splitting by store would put clips of a task in both train and eval, which measures
    memorisation rather than generalisation -- the failure the whole held-out protocol exists
    to avoid.
    """
    task_ids = [0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7]
    heldout = split_tasks(task_ids, 0.5)
    by_task: dict[int, set[bool]] = {}
    for task, flag in zip(task_ids, heldout, strict=True):
        by_task.setdefault(task, set()).add(bool(flag))
    assert all(len(sides) == 1 for sides in by_task.values()), "a task straddled the split"


def test_split_is_deterministic_across_calls():
    """Every rank and every requeue must agree without communicating."""
    task_ids = list(range(37))
    first = split_tasks(task_ids, 0.2)
    for _ in range(3):
        np.testing.assert_array_equal(split_tasks(task_ids, 0.2), first)


def test_split_reserves_the_requested_fraction_of_tasks():
    task_ids = [i // 3 for i in range(150)]  # 50 tasks, 3 stores each
    heldout = split_tasks(task_ids, 0.2)
    held_tasks = {t for t, flag in zip(task_ids, heldout, strict=True) if flag}
    assert len(held_tasks) == 10


@pytest.mark.parametrize("frac", [0.0, 1.0])
def test_degenerate_fractions_are_all_or_nothing(frac):
    task_ids = list(range(10))
    heldout = split_tasks(task_ids, frac)
    assert heldout.all() if frac == 1.0 else not heldout.any()


# --------------------------------------------------------------------------------------
# The fixed evaluation set
# --------------------------------------------------------------------------------------


def test_eval_set_is_identical_every_time_it_is_built():
    """The whole point: a change in the metric must be a change in the head.

    Resampling each eval made the number track which domain got drawn -- one run reported
    4.49, 4.30, 0.98, 1.95, 0.78, 2.54, 2.73 while its weights moved slowly and smoothly.
    """
    entries = np.stack([np.repeat(np.arange(20), 10), np.arange(200), np.ones(200), np.zeros(200)], axis=1)
    task_of_store = np.arange(20) % 5
    dataset = fake_dataset(entries)

    first = fixed_eval_indices(dataset, task_of_store, per_task=4)
    for _ in range(3):
        assert fixed_eval_indices(dataset, task_of_store, per_task=4) == first


def test_eval_set_is_balanced_across_tasks():
    """Unbalanced picks let one crowded task dominate top-1 and mask everything else."""
    # Task 0 has ten times the clips of the others; the eval set must not reflect that.
    store_of_clip = np.concatenate([np.zeros(500, int), np.repeat(np.arange(1, 5), 50)])
    entries = np.stack(
        [store_of_clip, np.arange(len(store_of_clip)), np.ones(len(store_of_clip)), np.zeros(len(store_of_clip))],
        axis=1,
    )
    task_of_store = np.arange(5)

    chosen = fixed_eval_indices(fake_dataset(entries), task_of_store, per_task=8)
    counts = np.bincount(task_of_store[store_of_clip[chosen]], minlength=5)
    assert counts.tolist() == [8, 8, 8, 8, 8]


def test_eval_set_takes_everything_from_a_task_with_too_few_clips():
    store_of_clip = np.concatenate([np.zeros(3, int), np.ones(20, int)])
    entries = np.stack([store_of_clip, np.arange(23), np.ones(23), np.zeros(23)], axis=1)
    chosen = fixed_eval_indices(fake_dataset(entries), np.arange(2), per_task=8)
    counts = np.bincount(np.arange(2)[store_of_clip[chosen]], minlength=2)
    assert counts.tolist() == [3, 8]


def test_eval_set_spreads_picks_rather_than_taking_a_prefix():
    """Consecutive clips overlap heavily in time; a prefix would sample one moment of one
    episode and call it a task."""
    entries = np.stack([np.zeros(100, int), np.arange(100), np.ones(100), np.zeros(100)], axis=1)
    chosen = fixed_eval_indices(fake_dataset(entries), np.zeros(1, dtype=int), per_task=5)
    assert chosen == [0, 24, 49, 74, 99]
