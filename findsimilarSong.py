import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

def find_similar_songs(clustered_csv, test_pca_vector, language, top_n=5):
    """
    根據 test 向量找出同 cluster 且同語言中最相似的歌曲。
    """
    # 載入 GMM 模型與 PCA 資料
    with open("./model/gmm_model.pkl", "rb") as f:
        gmm_model = pickle.load(f)
    with open("./data/X_train_pca.pkl", "rb") as f:
        X_train_pca = pickle.load(f)

    # 載入聚類結果 CSV
    df = pd.read_csv(clustered_csv)

    # 預測該歌曲所屬 cluster
    target_cluster = gmm_model.predict([test_pca_vector])[0]

    if(language == "none"):
        # 如果沒有指定語言，則不篩選語言
        filtered_df = df[df["Cluster"] == target_cluster]
    else:
        # 如果有指定語言，則篩選語言
        filtered_df = df[(df["Cluster"] == target_cluster) & (df["Language"] == language)]

    if filtered_df.empty:
        print(f"⚠️ No songs found in cluster {target_cluster} with language '{language}'")
        return []

    filtered_idx = filtered_df.index

    # 計算 cosine 相似度
    similarities = cosine_similarity([test_pca_vector], X_train_pca[filtered_idx])[0]

    # 取 top_n 最相似的歌曲
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_song_ids = df.iloc[filtered_idx[top_indices], 0].values
    top_scores = similarities[top_indices]

    return list(zip(top_song_ids, top_scores))
