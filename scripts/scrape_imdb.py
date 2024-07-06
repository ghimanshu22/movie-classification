import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_imdb_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    movie_data = []
    for movie in soup.find_all('div', class_='lister-item mode-advanced'):
        title = movie.h3.a.text
        genre = movie.find('span', class_='genre').text.strip()
        description = movie.find_all('p', class_='text-muted')[1].text.strip()
        poster_url = movie.find('img')['loadlate']
        
        movie_data.append({
            'title': title,
            'genre': genre,
            'description': description,
            'poster_url': poster_url
        })
    
    return movie_data

url = 'https://www.imdb.com/search/title/?genres=drama'
movies = scrape_imdb_data(url)

# Save to CSV
data = pd.DataFrame(movies)
data.to_csv('data/movie_data.csv', index=False)
