import os
from urllib.request import urlretrieve
import pandas as pd

def download_posters(movies, download_dir='data/posters'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for movie in movies:
        poster_url = movie['poster_url']
        poster_path = os.path.join(download_dir, f"{movie['title'].replace(' ', '_')}.jpg")
        urlretrieve(poster_url, poster_path)
        movie['poster_path'] = poster_path

movies = pd.read_csv('data/movie_data.csv').to_dict('records')
download_posters(movies)

# Update the CSV file with poster paths
data = pd.DataFrame(movies)
data.to_csv('data/movie_data_with_posters.csv', index=False)
