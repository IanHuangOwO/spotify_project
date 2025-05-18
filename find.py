import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

def find_similar_songs(clustered_csv,  test_pca_vector, target_language="en",top_n=5):
    
    with open("./assets/model/kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
        
    with open("./assets/data/X_train_pca.pkl", "rb") as f:
        X_train_pca = pickle.load(f)
    
    df = pd.read_csv(clustered_csv)
    
    # song_ids = df.iloc[:, 0].values
    # clusters = df['Cluster'].values
    
    target_cluster = kmeans_model.predict([test_pca_vector])[0]
    # 載入
    

    # same_cluster_idx = df[df['Cluster'] == target_cluster].index
    filtered_df = df[(df['Cluster'] == target_cluster) & (df['Language'] != target_language)]
    filtered_idx = filtered_df.index

    if len(filtered_idx) == 0:
        print("⚠️ 沒有找到不同語言的歌曲")
        return []
    similarities = cosine_similarity([test_pca_vector], X_train_pca[filtered_idx])[0]

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_song_ids = df.iloc[filtered_idx[top_indices], 0].values
    top_scores = similarities[top_indices]

    return list(zip(top_song_ids, top_scores))
