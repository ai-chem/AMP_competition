"""Universal syntactic filters for peptide sequences (length, alphabet, duplicates, reference matches)."""

from __future__ import annotations

import argparse
import json

import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

CANONICAL_20_AA = "ACDEFGHIKLMNPQRSTVWY"

class SyntacticFilter:
    """ Complex syntactic validation filter.
    
    Applies in order:
    1. Length filter (8-50 residues)
    2. Alphabet filter (only standard 20 AAs)
    3. Deduplication (remove duplicates)
    4. Exact reference match filter"""

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 50,
        allowed_alphabet: str = CANONICAL_20_AA,
        reference_sequences: Iterable[SeqRecord] | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_alphabet = frozenset(allowed_alphabet.upper())
        self._regex = re.compile(f"^[{re.escape(''.join(sorted(self.allowed_alphabet)))}]+$")

        self.reference_set: frozenset[str] = (
            frozenset(str(rec.seq).upper().strip() for rec in reference_sequences)
            if reference_sequences is not None
            else frozenset()
        )

    @classmethod
    def from_reference_fasta(
        cls,
        reference_fasta_path: str | Path | None,
        min_length: int = 8,
        max_length: int = 50,
    ) -> SyntacticFilter:
        ref_records = None
        if reference_fasta_path:
            p = Path(reference_fasta_path)
            if p.is_file():
                ref_records = list(SeqIO.parse(p, "fasta"))
            else:
                print(f"WARNING: Reference file {p} not found.", file=sys.stderr)
        return cls(min_length=min_length, max_length=max_length, reference_sequences=ref_records)

    def filter_records(
        self,
        records: Iterable[SeqRecord],
    ) -> tuple[list[SeqRecord], dict[str, Any]]:
        passed: list[SeqRecord] = []
        seen: set[str] = set()

        stats = {
            "total_input": 0,
            "passed": 0,
            "rejected_length": 0,
            "rejected_alphabet": 0,
            "rejected_duplicate": 0,
            "rejected_exact_reference": 0,
        }

        min_len = self.min_length
        max_len = self.max_length
        match = self._regex.match
        ref_set = self.reference_set

        for record in records:
            stats["total_input"] += 1

            seq_str = str(record.seq).upper().strip()
            length = len(seq_str)

            # 1. Length filter 
            if not (min_len <= length <= max_len):
                stats["rejected_length"] += 1
                continue

            # 2. Alphabet filter 
            if not match(seq_str):
                stats["rejected_alphabet"] += 1
                continue

            # 3. Deduplication 
            if seq_str in seen:
                stats["rejected_duplicate"] += 1
                continue

            # 4. Exact reference match filter
            if ref_set and (seq_str in ref_set):
                stats["rejected_exact_reference"] += 1
                continue

            seen.add(seq_str)
            passed.append(record)

        stats["passed"] = len(passed)
        stats["attrition_rate"] = round(
            (stats["total_input"] - stats["passed"]) / max(stats["total_input"], 1), 4
        )
        return passed, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply syntactic filters to a peptide FASTA pool.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input FASTA")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output filtered FASTA")
    parser.add_argument("-r", "--reference", type=Path, default=None, help="Reference FASTA")
    parser.add_argument("--min-len", type=int, default=8, help="Min length (default: 8)")
    parser.add_argument("--max-len", type=int, default=50, help="Max length (default: 50)")
    parser.add_argument("--stats-out", type=Path, default=None, help="Optional JSON path for stats")

    args = parser.parse_args()

    filter_engine = SyntacticFilter.from_reference_fasta(
        reference_fasta_path=args.reference,
        min_length=args.min_len,
        max_length=args.max_len,
    )

    print(f"Filtering {args.input}...", file=sys.stderr)
    passed, stats = filter_engine.filter_records(SeqIO.parse(args.input, "fasta"))
    
    SeqIO.write(passed, args.output, "fasta")

    print(f"Results: {json.dumps(stats, indent=2)}", file=sys.stderr)
    print(f"Saved {len(passed)} valid sequences to {args.output}", file=sys.stderr)

    if args.stats_out:
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()