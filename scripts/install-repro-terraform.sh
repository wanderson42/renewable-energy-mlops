#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=1.16.5
# Official SHA256SUMS entry for terraform_1.16.5_linux_amd64.zip.
checksum=2bc2fcfff033265c9e02ca0351f01794eb122f62a9b2a49a3294b9e49eaab5e4

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Este instalador fixa o binário Linux AMD64 usado no ensaio." >&2
  exit 1
fi
for tool in curl sha256sum python3; do
  command -v "$tool" >/dev/null || { echo "Ferramenta ausente: $tool" >&2; exit 1; }
done

mkdir -p "$project_root/.repro/bin"
download_dir="$(mktemp -d "$project_root/.repro/terraform-download.XXXXXX")"
trap 'rm -rf -- "$download_dir"' EXIT
curl --fail --silent --show-error --location --retry 2 --max-time 180 \
  "https://releases.hashicorp.com/terraform/${version}/terraform_${version}_linux_amd64.zip" \
  --output "$download_dir/terraform.zip"
printf '%s  %s\n' "$checksum" "$download_dir/terraform.zip" | sha256sum --check --status
python3 - "$download_dir" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

directory = Path(sys.argv[1])
with ZipFile(directory / "terraform.zip") as archive:
    (directory / "terraform").write_bytes(archive.read("terraform"))
PY
chmod 755 "$download_dir/terraform"
mv -- "$download_dir/terraform" "$project_root/.repro/bin/terraform"
"$project_root/.repro/bin/terraform" version
