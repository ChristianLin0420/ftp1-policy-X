#!/bin/bash
# Scheduler-backed status for Control V2 jobs. Silence from a log or marker is never treated as
# liveness; active state comes from squeue and terminal state comes from sacct.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 JOB_ID [JOB_ID ...]" >&2
  exit 2
fi

printf '%-12s %-12s %-28s %-20s %-12s %-12s %s\n' JOB_ID SOURCE NAME NODELIST STATE ELAPSED EXIT
for job_id in "$@"; do
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    printf '%-12s %-12s %-28s %-20s %-12s %-12s %s\n' "${job_id}" INVALID - - - - -
    continue
  fi
  queued="$(squeue -h -j "${job_id}" -o '%i|%j|%N|%T|%M' 2>/dev/null | head -1 || true)"
  if [[ -n "${queued}" ]]; then
    IFS='|' read -r id name node_list state elapsed <<< "${queued}"
    printf '%-12s %-12s %-28s %-20s %-12s %-12s %s\n' \
      "${id}" squeue "${name}" "${node_list}" "${state}" "${elapsed}" -
    continue
  fi
  accounted="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,JobName,NodeList,State,Elapsed,ExitCode \
    --parsable2 2>/dev/null | awk -F'|' -v id="${job_id}" '$1 == id {print; exit}' || true)"
  if [[ -n "${accounted}" ]]; then
    IFS='|' read -r id name node_list state elapsed exit_code _ <<< "${accounted}"
    printf '%-12s %-12s %-28s %-20s %-12s %-12s %s\n' \
      "${id}" sacct "${name}" "${node_list}" "${state}" "${elapsed}" "${exit_code}"
  else
    printf '%-12s %-12s %-28s %-20s %-12s %-12s %s\n' "${job_id}" UNKNOWN - - - - -
  fi
done
