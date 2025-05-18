import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import find as find

def get_song_API(target_features: list, target_language: str):
    """
    根據歌曲 ID 取得該首歌的特徵向量
    :param song_id: 歌曲 ID
    :return: 特徵向量
    """

    # 載入 scaler
    with open("./assets/model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 載入 PCA 模型
    with open("./assets/model/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
        
    features_scaled = scaler.transform(target_features)
    test_pca_vector = pca.transform(features_scaled)[0]

    # 找相似歌曲（在同 cluster 裡）
    print("正在尋找相似歌曲...")
    positiveResult = find.find_similar_songs("./assets/data/fliter_data_train.csv",  test_pca_vector=test_pca_vector, target_language=target_language,top_n=10)

    # for sid, score in result:
    #     print(f"🎵 Song ID: {sid}, Similarity: {score:.4f}")
    return positiveResult
