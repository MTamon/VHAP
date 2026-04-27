#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# VHAP third-party asset downloader.
#
# Downloads the FLAME assets that can be fetched with FLAME credentials and
# places them at the paths expected by vhap/model/flame.py.
#
# Usage:
#   bash download_assets.sh
#   bash download_assets.sh --flame_user USER --flame_pass PASS
#   bash download_assets.sh --no_flame
#
# Required target files:
#   asset/flame/flame2023.pkl
#   asset/flame/FLAME_masks.pkl
#
# Already tracked in this repository:
#   asset/flame/head_template_mesh.obj
#   asset/flame/landmark_embedding_with_eyes.npy
#   asset/flame/tex_mean_painted.png
#   asset/flame/uv_masks.npz
# -----------------------------------------------------------------------------

set -euo pipefail

WITH_FLAME=1
FLAME_USER=""
FLAME_PASS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no_flame) WITH_FLAME=0; shift ;;
    --flame_user) FLAME_USER="$2"; shift 2 ;;
    --flame_pass) FLAME_PASS="$2"; shift 2 ;;
    -h|--help)
      awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
      exit 0
      ;;
    *) echo "[download_assets.sh] unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

say() { printf '\n\033[1;36m[download_assets.sh] %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m[download_assets.sh] WARN: %s\033[0m\n' "$*"; }

need_bin() {
  command -v "$1" >/dev/null 2>&1 || {
    warn "required binary '$1' is missing on PATH."
    return 1
  }
}

urle() {
  [[ "${1}" ]] || return 1
  local LANG=C i x
  for (( i = 0; i < ${#1}; i++ )); do
    x="${1:i:1}"
    [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
  done
  echo
}

install_first_match() {
  local root="$1"
  local pattern="$2"
  local dest="$3"
  local found

  found="$(find "${root}" -type f -name "${pattern}" | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    cp -f "${found}" "${dest}"
    say "Installed ${dest}"
  else
    warn "Could not find ${pattern} under ${root}"
  fi
}

need_bin wget || exit 2
need_bin unzip || exit 2

mkdir -p asset/flame

FLAME_2023_PKL="asset/flame/flame2023.pkl"
FLAME_MASKS_PKL="asset/flame/FLAME_masks.pkl"

ensure_flame_credentials() {
  if [[ -z "${FLAME_USER}" ]]; then
    read -r -p "Username (FLAME): " FLAME_USER
  fi
  if [[ -z "${FLAME_PASS}" ]]; then
    read -r -s -p "Password (FLAME): " FLAME_PASS
    echo
  fi
}

download_flame_file() {
  local sfile="$1"
  local out="$2"

  ensure_flame_credentials

  local user_enc pass_enc
  user_enc="$(urle "${FLAME_USER}")"
  pass_enc="$(urle "${FLAME_PASS}")"

  wget --post-data "username=${user_enc}&password=${pass_enc}" \
    "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=${sfile}&resume=1" \
    -O "${out}" --no-check-certificate --continue
}

valid_download() {
  local file="$1"
  local min_bytes="$2"
  [[ -s "${file}" ]] || return 1
  [[ "$(wc -c < "${file}")" -ge "${min_bytes}" ]] || return 1
  if file "${file}" | grep -qiE 'HTML|ASCII text'; then
    return 1
  fi
}

if [[ ${WITH_FLAME} -eq 1 ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT

  if [[ -f "${FLAME_2023_PKL}" ]]; then
    say "flame2023.pkl already present, skipping."
  else
    say "Downloading FLAME2023.zip from the FLAME website."
    download_flame_file "FLAME2023.zip" "${TMP_DIR}/FLAME2023.zip"
    valid_download "${TMP_DIR}/FLAME2023.zip" 1048576 || {
      warn "FLAME2023.zip download did not look valid. Check your FLAME credentials/license access."
      exit 1
    }
    unzip -o "${TMP_DIR}/FLAME2023.zip" -d "${TMP_DIR}/FLAME2023"
    install_first_match "${TMP_DIR}/FLAME2023" "flame2023.pkl" "${FLAME_2023_PKL}"
  fi

  if [[ -f "${FLAME_MASKS_PKL}" ]]; then
    say "FLAME_masks.pkl already present, skipping."
  else
    say "Downloading FLAME_masks.zip from the public MPI FLAME file server."
    wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' \
      -O "${TMP_DIR}/FLAME_masks.zip" --no-check-certificate --continue
    valid_download "${TMP_DIR}/FLAME_masks.zip" 1024 || {
      warn "FLAME_masks.zip download did not look valid."
      warn "Manual fallback: download FLAME vertex masks from https://flame.is.tue.mpg.de/download.php and place FLAME_masks.pkl at ${FLAME_MASKS_PKL}."
      exit 1
    }
    unzip -t "${TMP_DIR}/FLAME_masks.zip" >/dev/null
    unzip -o "${TMP_DIR}/FLAME_masks.zip" -d "${TMP_DIR}/FLAME_masks"
    install_first_match "${TMP_DIR}/FLAME_masks" "FLAME_masks.pkl" "${FLAME_MASKS_PKL}"
  fi
else
  say "--no_flame given; skipping FLAME credential-gated assets."
fi

say "Asset summary:"
for f in \
  "${FLAME_2023_PKL}" \
  "${FLAME_MASKS_PKL}" \
  asset/flame/head_template_mesh.obj \
  asset/flame/landmark_embedding_with_eyes.npy \
  asset/flame/tex_mean_painted.png \
  asset/flame/uv_masks.npz; do
  if [[ -f "${f}" ]]; then
    printf '  %-52s %s\n' "${f}" "$(du -h "${f}" | cut -f1)"
  else
    printf '  %-52s %s\n' "${f}" "MISSING"
  fi
done

if [[ ! -f "${FLAME_2023_PKL}" || ! -f "${FLAME_MASKS_PKL}" ]]; then
  warn "Missing FLAME assets remain. Download them from https://flame.is.tue.mpg.de/download.php and place them in asset/flame/."
fi
