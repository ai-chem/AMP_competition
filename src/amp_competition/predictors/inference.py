import logging
from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmModel

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def clean_sequence(seq: str) -> str:
    return str(seq).upper().replace("X", "A").replace("B", "N").replace("Z", "Q").replace("U", "C").replace("O", "K")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AMPRanker:
    def __init__(
        self,
        models_dir: str | Path = "models",
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 64,
        local_files_only: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.batch_size = batch_size

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Missing models directory: {self.models_dir}")

        self.booster = lgb.Booster(model_file=str(self.models_dir / "lgbm_cv_ranker.txt"))
        self.scaler = joblib.load(self.models_dir / "robust_scaler_cv.pkl")
        self.pca = joblib.load(self.models_dir / "pca_cv.pkl")
        self.physchem_cols = joblib.load(self.models_dir / "physchem_cols.pkl")

        self.device = select_device()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.esm_model = EsmModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            self.esm_model = EsmModel.from_pretrained(model_name, local_files_only=False).to(self.device)

        self.esm_model.eval()
        for p in self.esm_model.parameters():
            p.requires_grad = False

    def _extract_esm_embeddings(self, sequences: list[str]) -> np.ndarray:
        embeddings_list = []

        with torch.inference_mode():
            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = sequences[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_seqs, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.esm_model(**inputs)
                token_embeddings = outputs.last_hidden_state
                input_ids = inputs["input_ids"]

                residue_mask = (
                    (input_ids != self.tokenizer.cls_token_id) &
                    (input_ids != self.tokenizer.eos_token_id) &
                    (input_ids != self.tokenizer.pad_token_id)
                ).unsqueeze(-1).to(token_embeddings.dtype)

                sum_embeddings = torch.sum(token_embeddings * residue_mask, dim=1)
                sum_mask = torch.clamp(residue_mask.sum(dim=1), min=1e-9)
                mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings_list.append(mean_embeddings)

        return np.vstack(embeddings_list).astype(np.float32)

    def _extract_physchem_features(self, sequences: list[str]) -> pd.DataFrame:
        features = []
        for seq in sequences:
            clean_seq = clean_sequence(seq)
            pa = ProteinAnalysis(clean_seq)

            aa_counts = pa.count_amino_acids()
            pos_charge = aa_counts.get("K", 0) + aa_counts.get("R", 0) + aa_counts.get("H", 0)
            neg_charge = aa_counts.get("D", 0) + aa_counts.get("E", 0)

            feat_dict = {
                "seq_len": len(clean_seq),
                "mol_weight": pa.molecular_weight(),
                "isoelectric_point": pa.isoelectric_point(),
                "gravy": pa.gravy(),
                "aromaticity": pa.aromaticity(),
                "charge_at_ph7": pa.charge_at_pH(7.0),
                "charge_at_ph3": pa.charge_at_pH(3.0),
                "charge_at_ph5": pa.charge_at_pH(5.0),
                "charge_at_ph9": pa.charge_at_pH(9.0),
                "instability_index": pa.instability_index(),
                "aliphatic_index": (
                    100 * (aa_counts.get("A", 0) + 2.9 * aa_counts.get("V", 0) + 3.9 * (aa_counts.get("I", 0) + aa_counts.get("L", 0))) / len(clean_seq)
                ),
                "pos_charge_count": pos_charge,
                "neg_charge_count": neg_charge,
                "net_charge_raw": pos_charge - neg_charge,
                "pos_charge_ratio": pos_charge / len(clean_seq),
                "hydrophobic_ratio": sum(aa_counts.get(aa, 0) for aa in "AILMFWV") / len(clean_seq),
            }
            features.append(feat_dict)

        df_feat = pd.DataFrame(features)

        missing_cols = set(self.physchem_cols) - set(df_feat.columns)
        if missing_cols:
            logging.warning(
                f"Missing {len(missing_cols)} features expected by the model: {missing_cols}. "
                f"Filling with 0.0."
            )
            for mc in missing_cols:
                df_feat[mc] = 0.0

        return df_feat[self.physchem_cols].fillna(0)

    def predict(self, sequences: list[str] | pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(sequences, (pd.Series, np.ndarray)):
            sequences = sequences.tolist()

        seq_list = [clean_sequence(s) for s in sequences]
        if not seq_list:
            return np.array([], dtype=np.float32)

        E = self._extract_esm_embeddings(seq_list)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        E_pca = self.pca.transform(E_norm)

        df_physchem = self._extract_physchem_features(seq_list)
        P_scaled = self.scaler.transform(df_physchem.values.astype(np.float32))

        X = np.hstack([E_pca, P_scaled])
        assert X.shape[1] == self.booster.num_feature(), (
            f"Feature count mismatch: input has {X.shape[1]} features, model expects {self.booster.num_feature()}"
        )

        return self.booster.predict(X)