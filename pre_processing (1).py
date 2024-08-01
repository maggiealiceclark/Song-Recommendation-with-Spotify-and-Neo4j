import pandas as pd
import math
from scipy import stats

ALBUM_NAME = 'Is This It'

def main():
    df = pd.read_csv('spotify.csv')
    target_album_df = df[df['album_name'] == ALBUM_NAME]
    sample = df.sample(100) 
    res = pd.concat([sample, target_album_df]).drop_duplicates()

    features = ['energy', 'speechiness', 'valence', 'danceability', 'acousticness', 'liveness', 'tempo']
    edges = pd.DataFrame(columns=['track_id', 'track_artist_name', 'track_name', 'track_album_name', 'track_genre', 'track_popularity', 'target_track_id', 'target_track_artist_name','target_track_name', 'target_track_album_name', 'target_track_genre', 'target_track_popularity', 'similarity'])
   
    for idx_outer, track in res.iterrows():
       for idx_inner, target_track in res.iterrows():
          if track['track_id'] != target_track['track_id'] and target_track['album_name'] == 'Is This It':
            dist = euclidean_distance(track[features].to_list(), target_track[features].to_list())
            edges.loc[len(edges.index)] = [track['track_id'], track['artists'], track['track_name'], track['album_name'], track['track_genre'], track['popularity'], target_track['track_id'], target_track['artists'], target_track['track_name'], target_track['album_name'], target_track['track_genre'], target_track['popularity'], dist]

    edges = standardize_similarity(edges)
    edges.to_csv('similarity_score.csv')
# computes euclidian distance between two arrays.
def euclidean_distance(x, y):
  return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
# standardize the similarity score to be a score from 1-100.
def standardize_similarity(edges):
    edges['similarity'] = stats.zscore(edges['similarity'])
    edges['similarity'].update((edges['similarity'] - edges['similarity'].min()) / (edges['similarity'].max() - edges['similarity'].min()) * 100)
    edges['similarity'] = edges['similarity'].astype('int')
    return edges 

if __name__ == '__main__':
    main()