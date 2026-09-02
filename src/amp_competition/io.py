"""FASTA helpers used by generation and filtering modules."""

from pathlib import Path


def write_fasta(sequences: list[str], path: Path, prefix: str = "seq") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences, start=1):
            handle.write(f">{prefix}{index}\n{sequence}\n")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(parts))
            header = line[1:]
            parts = []
        else:
            parts.append(line.upper())
    if header is not None:
        headers.append(header)
        sequences.append("".join(parts))
    return headers, sequences
