import argparse
import logging
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    if len(y_true) < 2:
        return 1.0
    eff_k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:eff_k]
    gain = y_true[order].astype(float)
    discounts = np.log2(np.arange(2, eff_k + 2))
    dcg = np.sum(gain / discounts)
    ideal_order = np.argsort(y_true)[::-1][:eff_k]
    ideal_gain = y_true[ideal_order].astype(float)
    idcg = np.sum(ideal_gain / discounts)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

def train_ranker(data_path: Path, emb_path: Path, models_dir: Path):
    logging.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=['sequence', 'bacterium', 'relevance']).reset_index(drop=True)
    
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    logging.info(f"Loading embeddings: {emb_path}")
    E = np.load(emb_path).astype(np.float32)
    assert len(E) == len(df), f"Mismatch: E={len(E)}, df={len(df)}"

    meta_cols = {'sequence', 'bacterium', 'mean', 'pmic', 'relevance', 'cluster_id', 'split', 'unnamed: 0'}
    physchem_cols = [c for c in df.columns if c not in meta_cols]
    P = df[physchem_cols].fillna(0).values.astype(np.float32)
    y = np.clip(df["relevance"].values.astype(int), 0, None)
    
    clusters = df["cluster_id"].fillna(df["bacterium"]).astype(str).values if "cluster_id" in df.columns else df["bacterium"].astype(str).values
    strains = df["bacterium"].astype(str).values

    # 1. Инвариант геометрии ESM-2 (L2-нормализация сфер. пространства)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=64, random_state=42)
    E_pca = pca.fit_transform(E_norm)

    scaler = RobustScaler()
    P_scaled = scaler.fit_transform(P)

    X = np.hstack([E_pca, P_scaled])

    # Динамический label_gain во избежание выхода за границы маппинга
    max_label = int(y.max())
    label_gain_list = [float(i) for i in range(max_label + 1)]

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "label_gain": label_gain_list,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1
    }

    # 2. Кросс-валидация по кластерам / штаммам
    logging.info("Starting 5-fold GroupKFold CV...")
    gkf = GroupKFold(n_splits=5)
    scores = []
    fold_idx = 1
    for train_idx, val_idx in gkf.split(X, y, groups=clusters):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        b_tr, b_va = strains[train_idx], strains[val_idx]

        sort_idx = np.argsort(b_tr)
        X_tr_s, y_tr_s, b_tr_s = X_tr[sort_idx], y_tr[sort_idx], b_tr[sort_idx]
        _, counts = np.unique(b_tr_s, return_counts=True)

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(X_tr_s, y_tr_s, group=counts)

        preds = ranker.predict(X_va)
        fold_ndcgs = []
        for org in np.unique(b_va):
            mask = (b_va == org)
            if np.sum(mask) >= 2:
                score = calculate_ndcg_at_k(y_va[mask], preds[mask], k=10)
                fold_ndcgs.append(score)
        if fold_ndcgs:
            mean_f = np.mean(fold_ndcgs)
            scores.append(mean_f)
            logging.info(f"Fold {fold_idx} NDCG@10: {mean_f:.4f} (organisms: {len(np.unique(b_va))})")
        fold_idx += 1

    logging.info(f"\n---> CV Overall Mean NDCG@10: {np.mean(scores):.4f}")

    # 3. Финальный пересчет на полном датасете и сохранение артефактов
    logging.info("Fitting final ranker on full dataset...")
    sort_idx = np.argsort(strains)
    ranker = lgb.LGBMRanker(**params)
    _, counts = np.unique(strains[sort_idx], return_counts=True)
    ranker.fit(X[sort_idx], y[sort_idx], group=counts)

    models_dir.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(models_dir / "lgbm_cv_ranker.txt"))
    joblib.dump(scaler, models_dir / "robust_scaler_cv.pkl")
    joblib.dump(pca, models_dir / "pca_cv.pkl")
    joblib.dump(physchem_cols, models_dir / "physchem_cols.pkl")
    logging.info("Clean training artifacts saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/enriched_ranked_dataset.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/features_esm2.npy"))
    parser.add_argument("--models_dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train_ranker(args.data, args.embeddings, args.models_dir)