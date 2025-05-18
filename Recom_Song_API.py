import pickle
import findsimilarSong

def get_song_API(features_df, language: str = "none"):
    """
    根據歌曲 ID 取得該首歌的特徵向量
    :param song_id: 歌曲 ID
    :return: 特徵向量
    """
    # # 載入 test 資料
    # test_df = pd.read_csv("./data/fliter_data_test.csv")
    # country_df = pd.read_csv("./data/universe_avg_data.csv")

    # # 提取特徵欄位
    # feature_columns = ["danceability", "energy", "loudness", "speechiness",
    #             "acousticness", "instrumentalness", "liveness", "valence",
    #             "tempo", "duration_ms","key_x", "key_y"]

    # 載入 scaler
    with open("./assets/model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 載入 PCA 模型
    with open("./assets/model/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
        
    print("模型載入完成")
    
    # # 根據 ID 找出該首歌在 test_df 中的 row
    # row = test_df[test_df.iloc[:, 0] == target_song_id]  # 假設 ID 是在第一欄
    
    # country_row = country_df[country_df["country"] == country]

    # if row.empty:
    #     print(f"❌ Song ID {target_song_id} not found in test set.")
    # if country_row.empty:
    #     print(f"❌ Country {country} not found in country set.")
    # else:
    # 取得該歌的特徵向量 → 標準化 → PCA
    # song_features = row[feature_columns].values.astype(float)
    # country_features = country_row[feature_columns].values.astype(float)
    # features = 0.8 * song_features + 0.2 * country_features
    
    # features_df = pd.DataFrame(features, columns=feature_columns)  # 🟢 加這行
    features_scaled = scaler.transform(features_df)
    test_pca_vector = pca.transform(features_scaled)[0]

    # 找相似歌曲（在同 cluster 裡）
    print("正在尋找相似歌曲...")
    positiveResult = findsimilarSong.find_similar_songs(
        clustered_csv="./assets/data/clustered_gmm_result.csv",
        test_pca_vector=test_pca_vector,
        language=language,
        top_n=10
    )
    
    return positiveResult
