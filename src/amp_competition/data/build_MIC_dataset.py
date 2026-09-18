import argparse
import logging
import re
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import Levenshtein
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GRAMPA_URL = "https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/master/data/grampa.csv"

def download_grampa(input_path: Path) -> None:
    """Ensures raw dataset exists locally, downloading it if missing or invalid."""
    if input_path.exists() and input_path.stat().st_size > 1000:
        logging.info(f"Raw dataset validated at {input_path}. Skipping download.")
        return

    input_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading raw dataset from {GRAMPA_URL}...")

    try:
        response = requests.get(GRAMPA_URL, timeout=30)
        response.raise_for_status()
        input_path.write_bytes(response.content)
        size_mb = input_path.stat().st_size / (1024 * 1024)
        logging.info(f"Successfully downloaded {input_path.name} ({size_mb:.2f} MB)")
    except requests.RequestException as e:
        logging.error(f"Network error during download: {e}")
        if not input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {input_path} and automatic download failed. "
                f"Manually save {GRAMPA_URL} to {input_path.absolute()}"
            ) from e

def clean_mic(value: str) -> float:
    """Extracts numeric MIC value, ignoring inequality operators."""
    match = re.search(r"(\d+(\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan

def normalize_bacterium_name(name: str) -> str:
    """Norm/cleans organism names to merge typos like baumanii/baumannii."""
    if not isinstance(name, str):
        return "unknown"
    s = name.strip()
    # Стандартизация опечаток в Acinetobacter baumannii
    s = re.sub(r'^(A\.\s*|A\s+|Acinetobacter\s+)?baumanii.*', 'Acinetobacter baumannii', s, flags=re.IGNORECASE)
    s = re.sub(r'^(A\.\s*|A\s+|Acinetobacter\s+)?baumannii.*', 'Acinetobacter baumannii', s, flags=re.IGNORECASE)
    s = re.sub(r'^(A\.\s*|A\s+|Alternaria\s+)?brassicicola.*', 'Alternaria brassicicola', s, flags=re.IGNORECASE)
    s = re.sub(r'^(A\.\s*|Alternaria\s+)?brassicola.*', 'Alternaria brassicicola', s, flags=re.IGNORECASE)
    # Базовая подчистка лишних пробелов
    s = re.sub(r'\s+', ' ', s)
    return s.title() if len(s.split()) <= 2 else s

def greedy_clustering(sequences: list[str], threshold: float = 0.75) -> dict[str, int]:
    """O(N^2) greedy clustering using Levenshtein distance for sequence isolation."""
    clusters = {}
    centroids = []
    sorted_seqs = sorted(sequences, key=len, reverse=True)
    
    for seq in tqdm(sorted_seqs, desc="Clustering sequences"):
        assigned = False
        seq_len = len(seq)
        for i, center in enumerate(centroids):
            if abs(seq_len - len(center)) / max(seq_len, len(center)) > (1 - threshold):
                continue
                
            sim = 1.0 - (Levenshtein.distance(seq, center) / max(seq_len, len(center)))
            if sim >= threshold:
                clusters[seq] = i
                assigned = True
                break
                
        if not assigned:
            clusters[seq] = len(centroids)
            centroids.append(seq)
            
    return clusters

def run_pipeline(input_path: Path, output_path: Path, min_strain_samples: int = 5):
    # 1. Automatic dataset acquisition
    download_grampa(input_path)

    # 2. Data Loading & Sanitization
    logging.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    
    if 'target' in df.columns:
        df.rename(columns={'target': 'bacterium'}, inplace=True)
        
    df = df.dropna(subset=['sequence', 'bacterium', 'value'])
    df['sequence'] = df['sequence'].str.upper()
    
    # Нормализация таксономии
    df['bacterium'] = df['bacterium'].apply(normalize_bacterium_name)

    initial_len = len(df)
    df = df[df['sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$')]
    df = df[(df['sequence'].str.len() >= 8) & (df['sequence'].str.len() <= 100)]
    
    df['mic_clean'] = df['value'].apply(clean_mic)
    df = df.dropna(subset=['mic_clean'])
    df['pmic'] = -np.log10(df['mic_clean'] + 1e-6)
    
    # 3. Aggregation & Scoring
    logging.info("Aggregating duplicates by sequence and bacterium...")
    agg_df = df.groupby(['sequence', 'bacterium'])['pmic'].mean().reset_index()
    
    # Фильтрация штаммов с маленьким пулом кандидатов
    strain_counts = agg_df['bacterium'].value_counts()
    valid_strains = strain_counts[strain_counts >= min_strain_samples].index
    before_filter_strains = len(agg_df)
    agg_df = agg_df[agg_df['bacterium'].isin(valid_strains)].reset_index(drop=True)
    logging.info(f"Filtered out {before_filter_strains - len(agg_df)} records belonging to strains with < {min_strain_samples} candidates.")

    p_min, p_max = agg_df['pmic'].min(), agg_df['pmic'].max()
    agg_df['relevance'] = np.clip(((agg_df['pmic'] - p_min) / (p_max - p_min) * 100), 0, 100).astype(int)
    
    # 4. Clustering (Leakage Prevention)
    logging.info("Executing sequence clustering...")
    unique_seqs = agg_df['sequence'].unique().tolist()
    cluster_map = greedy_clustering(unique_seqs, threshold=0.75)
    agg_df['cluster_id'] = agg_df['sequence'].map(cluster_map)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    
    logging.info(f"Dataset generated: {len(agg_df)} records, {len(set(cluster_map.values()))} clusters across {agg_df['bacterium'].nunique()} clean strains.")
    logging.info(f"Dropped {initial_len - len(df)} invalid/out-of-bounds sequences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/GRAMPA/grampa.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/enriched_ranked_dataset.csv"))
    parser.add_argument("--min_strain", type=int, default=5)
    args = parser.parse_args()
    run_pipeline(args.input, args.output, min_strain_samples=args.min_strain)