from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REPOSITORY_URL = "https://github.com/szczurek-lab/amp-challenge-2027"
DEFAULT_COMMIT = "5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e"
REFERENCE_PATH = "data/antibacterial.fasta"


def _raw_url(commit: str) -> str:
    return (
        "https://raw.githubusercontent.com/szczurek-lab/amp-challenge-2027/"
        f"{commit}/{REFERENCE_PATH}"
    )


def _count_fasta_records(contents: bytes) -> int:
    return sum(line.startswith(b">") for line in contents.splitlines())


def fetch_reference(commit: str, output: Path, manifest: Path, force: bool) -> None:
    url = _raw_url(commit)
    request = Request(url, headers={"User-Agent": "AMP-competition-reference-fetcher/1"})
    with urlopen(request, timeout=60) as response:
        contents = response.read()

    record_count = _count_fasta_records(contents)
    if not contents or record_count == 0:
        raise ValueError(f"Downloaded file from {url} is empty or is not a FASTA file.")

    digest = hashlib.sha256(contents).hexdigest()
    if output.exists() and output.read_bytes() != contents and not force:
        raise FileExistsError(
            f"{output} already exists and differs from the downloaded reference. "
            "Review it or rerun with --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(contents)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "source_repository": REPOSITORY_URL,
        "source_commit": commit,
        "source_path": REFERENCE_PATH,
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "local_path": str(output),
        "sha256": digest,
        "size_bytes": len(contents),
        "fasta_record_count": record_count,
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Saved {record_count} FASTA records to {output}")
    print(f"Pinned source commit: {commit}")
    print(f"SHA-256: {digest}")
    print(f"Manifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=DEFAULT_COMMIT, help="Full organizer Git commit SHA.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/antibacterial.fasta"),
        help="Destination for the unmodified organizer FASTA.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external/antibacterial.manifest.json"),
        help="Destination for provenance and integrity metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing, different output FASTA.",
    )
    args = parser.parse_args()
    fetch_reference(args.commit, args.output, args.manifest, args.force)


if __name__ == "__main__":
    main()
