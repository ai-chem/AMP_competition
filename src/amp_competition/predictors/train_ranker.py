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
    return float(dcg / idcg) if idcg > 0 else 0.0


def train_ranker(data_path: Path, emb_path: Path, models_dir: Path):
    logging.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=['sequence', 'bacterium', 'relevance']).reset_index(drop=True)

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")

    logging.info(f"Loading embeddings: {emb_path}")
    E = np.load(emb_path).astype(np.float32)
    assert len(E) == len(df), f"Mismatch: E={len(E)}, df={len(df)}"

    meta_cols = {'sequence', 'bacterium', 'mean', 'pmic', 'relevance', 'cluster_id', 'split', 'unnamed: 0'}
    physchem_cols = [c for c in df.columns if c.lower() not in meta_cols]

    P = df[physchem_cols].fillna(0).values.astype(np.float32)
    y = np.clip(df["relevance"].values.astype(int), 0, None)

    clusters = df["cluster_id"].fillna(df["bacterium"]).astype(str).values if "cluster_id" in df.columns else df["bacterium"].astype(str).values
    strains = df["bacterium"].astype(str).values

    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    max_label = int(y.max())
    label_gain_list = [float(i) for i in range(max_label + 1)]

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
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

    logging.info("Starting 5-fold GroupKFold CV...")
    gkf = GroupKFold(n_splits=5)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(E_norm, y, groups=clusters), 1):
        fold_pca = PCA(n_components=64, random_state=42)
        E_tr_pca = fold_pca.fit_transform(E_norm[train_idx])
        E_va_pca = fold_pca.transform(E_norm[val_idx])

        fold_scaler = RobustScaler()
        P_tr_sc = fold_scaler.fit_transform(P[train_idx])
        P_va_sc = fold_scaler.transform(P[val_idx])

        X_tr = np.hstack([E_tr_pca, P_tr_sc])
        X_va = np.hstack([E_va_pca, P_va_sc])

        y_tr, y_va = y[train_idx], y[val_idx]
        b_tr, b_va = strains[train_idx], strains[val_idx]

        sort_tr = np.argsort(b_tr)
        X_tr_s, y_tr_s, b_tr_s = X_tr[sort_tr], y_tr[sort_tr], b_tr[sort_tr]
        _, counts_tr = np.unique(b_tr_s, return_counts=True)

        sort_va = np.argsort(b_va)
        X_va_s, y_va_s, b_va_s = X_va[sort_va], y_va[sort_va], b_va[sort_va]
        _, counts_va = np.unique(b_va_s, return_counts=True)

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            X_tr_s,
            y_tr_s,
            group=counts_tr,
            eval_X=X_va_s,
            eval_y=y_va_s,
            eval_group=[counts_va],
            eval_at=[10],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

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

    logging.info(f"CV Overall Mean NDCG@10: {np.mean(scores):.4f}")

    logging.info("Fitting final transformers and ranker on full dataset...")
    final_pca = PCA(n_components=64, random_state=42)
    E_full_pca = final_pca.fit_transform(E_norm)

    final_scaler = RobustScaler()
    P_full_sc = final_scaler.fit_transform(P)

    X_full = np.hstack([E_full_pca, P_full_sc])

    sort_full = np.argsort(strains)
    X_full_s = X_full[sort_full]
    y_full_s = y[sort_full]
    _, counts_full = np.unique(strains[sort_full], return_counts=True)

    final_ranker = lgb.LGBMRanker(**params)
    final_ranker.fit(X_full_s, y_full_s, group=counts_full)

    models_dir.mkdir(parents=True, exist_ok=True)
    final_ranker.booster_.save_model(str(models_dir / "lgbm_cv_ranker.txt"))
    joblib.dump(final_scaler, models_dir / "robust_scaler_cv.pkl")
    joblib.dump(final_pca, models_dir / "pca_cv.pkl")
    joblib.dump(physchem_cols, models_dir / "physchem_cols.pkl")
    logging.info("Clean training artifacts saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/enriched_ranked_dataset.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/features_esm2.npy"))
    parser.add_argument("--models_dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train_ranker(args.data, args.embeddings, args.models_dir)