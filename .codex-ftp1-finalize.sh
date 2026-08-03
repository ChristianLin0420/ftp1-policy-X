#!/usr/bin/env bash
set -euo pipefail

repo_root="/lustre/fsw/portfolios/edgeai/users/chrislin/projects/ftp1-policy-X"
dataset_root="$repo_root/FTP-1-Dataset"
active_download_pid="99999999"
log_file="/tmp/ftp1-download-extract.log"

exec >>"$log_file" 2>&1
echo "[$(date --iso-8601=seconds)] Waiting for initial Hugging Face download (PID $active_download_pid)"
while kill -0 "$active_download_pid" 2>/dev/null; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] Running resumable Hugging Face completion pass"
HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=1800 \
  hf download MJJJJ1064/FTP-1-Dataset --repo-type dataset --max-workers 8 --local-dir "$dataset_root"

mapfile -t archive_dirs < <(
  find "$dataset_root" -mindepth 1 -maxdepth 1 -type d ! -name '.cache' ! -name '.extracting_*' ! -name '.archives_*' -print | sort
)
if [[ ${#archive_dirs[@]} -ne 15 ]]; then
  echo "[ERROR] Expected 15 dataset archive directories, found ${#archive_dirs[@]}"
  exit 1
fi

for source_dir in "${archive_dirs[@]}"; do
  domain="$(basename "$source_dir")"
  staging_dir="$dataset_root/.extracting_$domain"
  archive_backup="$dataset_root/.archives_$domain"

  if [[ -e "$staging_dir" || -e "$archive_backup" ]]; then
    echo "[ERROR] Refusing to overwrite staging path for $domain"
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] Extracting $domain"
  bash "$repo_root/scripts/stage_ftp1_archive.sh" "$source_dir" "$staging_dir"

  shopt -s dotglob nullglob
  extracted_entries=("$staging_dir"/*)
  shopt -u dotglob nullglob
  if [[ ${#extracted_entries[@]} -eq 0 ]]; then
    echo "[ERROR] Extraction produced no files for $domain"
    exit 1
  fi

  mv "$source_dir" "$archive_backup"
  mv "$staging_dir" "$source_dir"
  rm -rf -- "$archive_backup"
  echo "[$(date --iso-8601=seconds)] Extracted $domain and removed its compressed archive files"
done

remaining_archives="$(find "$dataset_root" -type f \( -name '*.tar' -o -name '*.tar.part-*' -o -name '*.tgz' -o -name '*.tar.gz' -o -name '*.zip' \) -print -quit)"
if [[ -n "$remaining_archives" ]]; then
  echo "[ERROR] Compressed archive remains: $remaining_archives"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] COMPLETE: all FTP-1 datasets downloaded, extracted, and archives removed"
rm -- "$repo_root/.codex-ftp1-finalize.sh"
