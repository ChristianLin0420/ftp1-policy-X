#!/bin/bash
# Health readout for a MoT-JEPA pretraining log. Usage: bash watch.sh [logfile]
#
# Deliberately does NOT lead with loss. The loss minimises at collapse and rose 43% on probe2
# while effective rank doubled, so it cannot tell a healthy run from a collapsing one -- it sat
# flat at 0.81 through the entire tactile collapse of job 6562515. Rank and the student/teacher
# dispersion ratio are what moved.
set -uo pipefail
LOG="${1:-$(ls -t logs/motjepa-forecast100k.*.log 2>/dev/null | head -1)}"
[[ -f "$LOG" ]] || { echo "no log found: ${LOG:-<none>}" >&2; exit 1; }

python3 - "$LOG" <<'PY'
import ast, pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text(errors="ignore").splitlines()
probes, steps = {}, []
for line in text:
    m = re.search(r"INFO probes @(\d+): (\{.*\})", line)
    if m:
        try:
            probes[int(m[1])] = ast.literal_eval(m[2])
        except Exception:
            pass
    m = re.search(r"INFO step (\d+) loss ([0-9.]+) grad ([0-9.]+)", line)
    if m:
        steps.append((int(m[1]), float(m[2]), float(m[3])))
probes.pop(10, None)

print(f"log {sys.argv[1]}")
if steps:
    print(f"step {steps[-1][0]}  loss {steps[-1][1]:.3f} (not a health metric)")
    tail = [g for _, _, g in steps[-40:]]
    print(f"grad over last {len(tail)} logs: mean {sum(tail)/len(tail):.3f}  max {max(tail):.2f}   [want ~0.2 post-warmup, NOT climbing]")

print(f"\n{'step':>7} {'rankV':>6} {'rankT':>6} {'disp ratio':>11} {'retr':>6}   verdict")
for s in sorted(probes):
    v = probes[s]
    sd, td = v.get("student_dispersion", 0), v.get("teacher_dispersion", 0)
    ratio = sd / td if td else float("nan")
    # 0.90 is the alarm: probe3 read 0.93 two probes before it fell to 0.37, and 6562515 read
    # 0.89 five probes before 0.69. Below 0.90 the student is already leaving the teacher.
    flag = "OK" if ratio >= 0.90 else ("WATCH" if ratio >= 0.80 else "COLLAPSING -- stop the run")
    print(f"{s:>7} {v.get('rankme_video',0):6.0f} {v.get('rankme_tactile',0):6.0f} {ratio:11.2f} {v.get('retrieval_top1',0):6.3f}   {flag}")

print("\nreference: probe2 reached rankV ~270, rankT ~140 and held; ratio should sit near 0.95.")
PY
