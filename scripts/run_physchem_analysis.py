"""Physicochemical AMP vs putative non-AMP analysis.

Usage:
    uv sync --extra analysis
    PYTHONPATH=src python3 scripts/run_physchem_analysis.py --config physchem_analysis.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, load_config
from amp_competition.features.pipeline import run_physchem_analysis


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="physchem_analysis.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    physchem = config.get("physchem", {})

    parser = argparse.ArgumentParser(
        description="Physicochemical comparison of AMP vs putative non-AMP peptides.",
    )
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="YAML in configs/ or a path",
    )
    parser.add_argument(
        "--amp-fasta",
        type=Path,
        default=REPO_ROOT / physchem.get("amp_fasta", "data/external/antibacterial.fasta"),
        help="AMP/reference FASTA used as the positive dataset.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / physchem.get("out_dir", "outputs/physchem"),
        help="Output directory.",
    )
    parser.add_argument(
        "--downloaded-nonamp-fasta",
        type=Path,
        default=physchem.get("downloaded_nonamp_fasta"),
        help=(
            "Optional existing UniProt FASTA. If omitted, the pipeline downloads "
            "the dataset from UniProtKB."
        ),
    )
    parser.add_argument(
        "--unmatched-size",
        default=str(physchem.get("unmatched_size", "amp")),
        help=(
            'Size of the unmatched negative cohort: "amp" for the same size as '
            'the AMP set, "all" for all cleaned negatives, or an integer.'
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config.get("seed", DEFAULT_SEED)),
    )
    args = parser.parse_args()

    run_physchem_analysis(
        amp_fasta=args.amp_fasta,
        outdir=args.outdir,
        downloaded_nonamp_fasta=args.downloaded_nonamp_fasta,
        unmatched_size=args.unmatched_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
