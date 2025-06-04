import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from models import (
    cosine_similarity_numpy,
    PCANumpy,
    StandardScalerNumpy,
    GaussianMixtureNumpy,
)

_ASSETS_DIR = Path("./assets")
_MODEL_DIR = _ASSETS_DIR / "model"
_DATA_DIR = _ASSETS_DIR / "data"

_SCALER_PATH       = _MODEL_DIR / "scaler.pkl"
_PCA_PATH          = _MODEL_DIR / "pca_model.pkl"
_GMM_PATH          = _MODEL_DIR / "gmm_model.pkl"
_X_TRAIN_PCA_PATH  = _MODEL_DIR / "X_train_pca.pkl"
_CLUSTERED_CSV     = _DATA_DIR  / "clustered_gmm_result.csv"

class SongRecommender:
    def __init__(self):
        # Load (and cache in memory) all pickles + CSV
        self.scaler:       StandardScalerNumpy    = self._load_scaler()
        self.pca:          PCANumpy               = self._load_pca()
        self.gmm:          GaussianMixtureNumpy   = self._load_gmm()
        self.X_train_pca:  np.ndarray             = self._load_X_train_pca()
        self.df_clusters:  pd.DataFrame           = self._load_clustered_dataframe()

        # Verify that the DataFrame and X_train_pca align.
        if len(self.df_clusters) != self.X_train_pca.shape[0]:
            raise ValueError(
                f"clustered_gmm_result.csv has {len(self.df_clusters)} rows, "
                f"but X_train_pca.pkl has {self.X_train_pca.shape[0]} samples."
            )

    def get_similar_songs(
        self,
        song_id: int,
        features: np.ndarray,
        language: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Tuple[int, float]]:

        # Validate + reshape raw_features → 2D array.
        X_feat = np.asarray(features, dtype=float)
        if X_feat.ndim == 1:
            X_feat = X_feat[np.newaxis, :]
        elif X_feat.ndim != 2:
            raise ValueError(
                f"raw_features must be 1D or 2D; got shape {X_feat.shape}."
            )

        # Scale + PCA‐project
        X_scaled = self.scaler.transform(X_feat)
        pca_vector = self.pca.transform(X_scaled)[0]

        # Delegate to private helper to find top‐N in same cluster & language
        return self._find_candidates_in_same_cluster(
            song_id=song_id,
            test_pca_vector=pca_vector,
            language=language,
            top_n=top_n,
        )

    @staticmethod
    def _load_scaler() -> StandardScalerNumpy:
        with _SCALER_PATH.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_pca() -> PCANumpy:
        with _PCA_PATH.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_gmm() -> GaussianMixtureNumpy:
        with _GMM_PATH.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_X_train_pca() -> np.ndarray:
        with _X_TRAIN_PCA_PATH.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_clustered_dataframe() -> pd.DataFrame:
        df = pd.read_csv(_CLUSTERED_CSV)
        required_cols = {"id", "Cluster", "Language"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise KeyError(
                f"clustered_gmm_result.csv is missing columns: {missing}"
            )
        return df

    def _find_candidates_in_same_cluster(
        self,
        song_id: int,
        test_pca_vector: np.ndarray,
        language: Optional[str],
        top_n: int,
    ) -> List[Tuple[int, float]]:

        # Predict cluster for this single‐row vector
        test_2d = test_pca_vector[np.newaxis, :]
        predicted_cluster = self.gmm.predict(test_2d)[0]

        # Build a boolean mask on the DataFrame
        mask = (self.df_clusters["Cluster"] == predicted_cluster)
        if language:
            mask &= (self.df_clusters["Language"] == language)

        df_filtered = self.df_clusters.loc[mask].copy()

        # Exclude the query song if it’s present
        df_filtered = df_filtered[df_filtered["id"] != song_id]
        if df_filtered.empty:
            # No candidates in this cluster/language
            return []

        # Gather candidate indices
        candidate_idxs = df_filtered.index.to_numpy()
        candidates_pca = self.X_train_pca[candidate_idxs, :]

        # Compute cosine similarities
        sims = cosine_similarity_numpy(test_2d, candidates_pca)[0]

        # Pick top_n indices by descending similarity
        top_n = min(top_n, sims.size)
        top_candidate_idxs = np.argsort(sims)[::-1][:top_n]

        selected_rows = df_filtered.iloc[top_candidate_idxs]
        top_song_ids = selected_rows["id"].to_numpy()
        top_scores   = sims[top_candidate_idxs]

        return [(song, float(score)) for song, score in zip(top_song_ids, top_scores)]