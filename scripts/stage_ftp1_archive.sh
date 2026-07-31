#!/usr/bin/env bash
# Validate and extract one downloaded FTP-1 dataset archive without creating a second combined tar.
set -euo pipefail

usage() {
  echo "Usage: $0 [--check-only] <archive-file-or-parts-directory> <output-directory>" >&2
}

check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
  shift
fi
if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

source_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"
output_parent="$(dirname "$output_path")"

if [[ ! -e "$source_path" ]]; then
  echo "[ERROR] Archive source does not exist: $source_path" >&2
  exit 1
fi
if [[ -e "$output_path" ]]; then
  echo "[ERROR] Refusing to overwrite existing output: $output_path" >&2
  exit 1
fi

parts=()
if [[ -f "$source_path" ]]; then
  parts=("$source_path")
elif [[ -d "$source_path" ]]; then
  mapfile -t parts < <(find "$source_path" -maxdepth 1 -type f -name '*.tar.part-*' -print | sort)
  if [[ ${#parts[@]} -eq 0 ]]; then
    mapfile -t parts < <(find "$source_path" -maxdepth 1 -type f -name '*.tar' -print | sort)
  fi
else
  echo "[ERROR] Unsupported archive source: $source_path" >&2
  exit 1
fi
if [[ ${#parts[@]} -eq 0 ]]; then
  echo "[ERROR] No .tar or .tar.part-* files found under $source_path" >&2
  exit 1
fi

if [[ "${parts[0]}" == *.tar.part-* ]]; then
  expected=0
  for part in "${parts[@]}"; do
    suffix="${part##*.tar.part-}"
    if [[ ! "$suffix" =~ ^[0-9]+$ ]] || (( 10#$suffix != expected )); then
      printf '[ERROR] Split archive is not consecutive: expected part %04d, got %s\n' "$expected" "$part" >&2
      exit 1
    fi
    expected=$((expected + 1))
  done
elif [[ ${#parts[@]} -ne 1 ]]; then
  echo "[ERROR] Expected one .tar file, found ${#parts[@]}" >&2
  exit 1
fi

archive_bytes=0
for part in "${parts[@]}"; do
  part_bytes="$(stat -c '%s' "$part")"
  archive_bytes=$((archive_bytes + part_bytes))
done
available_bytes=$(( $(df --output=avail -B1 "$output_parent" | tail -n 1) ))

echo "[OK] ${#parts[@]} archive file(s), $archive_bytes bytes; destination has $available_bytes bytes free"
if (( available_bytes < archive_bytes * 2 )); then
  echo "[ERROR] Require at least 2x compressed size as extraction headroom" >&2
  exit 1
fi

if (( check_only )); then
  echo "[OK] Archive parts and storage headroom validated"
  exit 0
fi

mkdir -p "$output_parent"
staging_dir="$(mktemp -d "${output_parent}/.ftp1_extract_XXXXXX")"
cleanup() {
  if [[ -d "$staging_dir" ]]; then
    rm -rf -- "$staging_dir"
  fi
}
trap cleanup EXIT

if [[ "${parts[0]}" == *.tar.part-* ]]; then
  cat "${parts[@]}" | tar -xf - -C "$staging_dir"
else
  tar -xf "${parts[0]}" -C "$staging_dir"
fi

mv "$staging_dir" "$output_path"
trap - EXIT
echo "[OK] Extracted FTP-1 archive to $output_path"
