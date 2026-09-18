import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_embeddings(
    input_path: Path,
    output_path: Path,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 64,
) -> None:
    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    if "sequence" not in df.columns:
        raise KeyError("Input dataset must contain a 'sequence' column.")

    unique_seqs = df["sequence"].unique().tolist()
    device = select_device()
    logging.info(f"Initializing ESM-2 ({model_name}) on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings_map: dict[str, np.ndarray] = {}

    logging.info(f"Extracting features for {len(unique_seqs)} unique sequences...")
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_seqs), batch_size), desc="ESM-2 Batches"):
            batch_seqs = unique_seqs[i : i + batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True).to(
                device
            )

            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state

            # Mean pooling ignoring padding tokens
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            for seq, emb in zip(batch_seqs, mean_embeddings):
                embeddings_map[seq] = emb

    # Map back to full dataset order to guarantee row alignment
    logging.info("Aligning embeddings with dataset rows...")
    full_embeddings = np.array([embeddings_map[seq] for seq in df["sequence"]], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, full_embeddings)
    logging.info(f"Saved embeddings tensor {full_embeddings.shape} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/enriched_ranked_dataset.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features_esm2.npy"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    extract_embeddings(args.input, args.output, batch_size=args.batch_size)