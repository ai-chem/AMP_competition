import argparse
import logging
from pathlib import Path
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def clean_sequence(seq: str) -> str:
    """
    Очищает последовательность от нестандартных аминокислот для корректной работы Biopython.
    """
    return str(seq).upper().replace("X", "A").replace("B", "N").replace("Z", "Q").replace("U", "C").replace("O", "K")


def extract_physchem_features(input_path: Path, output_path: Path) -> None:
    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    if "sequence" not in df.columns:
        raise KeyError("Dataset must contain a 'sequence' column.")

    logging.info(f"Extracting physical-chemical features for {len(df)} sequences...")
    features = []
    
    for seq in df["sequence"]:
        clean_seq = clean_sequence(seq)
        try:
            pa = ProteinAnalysis(clean_seq)
            features.append({
                "seq_len": len(clean_seq),
                "mol_weight": pa.molecular_weight(),
                "isoelectric_point": pa.isoelectric_point(),
                "gravy": pa.gravy(),
                "aromaticity": pa.aromaticity(),
                "charge_at_ph7": pa.charge_at_pH(7.0)
            })
        except Exception as e:
            logging.warning(f"Failed to parse sequence {seq}: {e}. Assigning zero values.")
            features.append({
                "seq_len": len(clean_seq),
                "mol_weight": 0.0,
                "isoelectric_point": 0.0,
                "gravy": 0.0,
                "aromaticity": 0.0,
                "charge_at_ph7": 0.0
            })

    feat_df = pd.DataFrame(features, index=df.index)
    
    # Удаляем колонки, если скрипт запускается повторно по тому же файлу
    cols_to_drop = [c for c in feat_df.columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    out_df = pd.concat([df, feat_df], axis=1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logging.info(f"Saved dataset with physchem features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract physical-chemical features using Biopython.")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/processed/enriched_ranked_dataset.csv"),
        help="Path to input dataset."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/processed/enriched_ranked_dataset.csv"),
        help="Path to save output dataset (can overwrite input)."
    )
    args = parser.parse_args()

    extract_physchem_features(args.input, args.output)