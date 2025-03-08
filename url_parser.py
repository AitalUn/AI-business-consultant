import requests
from bs4 import BeautifulSoup

def get_article_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Ошибка загрузки страницы")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Извлечение заголовка статьи
    title_tag = soup.find('h1', class_='headline')
    title = title_tag.text.strip() if title_tag else 'Заголовок не найден'
    
    # Извлечение основного контента статьи
    content_div = soup.find('div', class_='article-body')
    paragraphs = content_div.find_all('p') if content_div else []
    content = '\n'.join([p.text.strip() for p in paragraphs])
    
    return title, content

if __name__ == "__main__":
    url = "https://www.express.co.uk/news/science/1323861/nasa-satellite-images-solar-flare-astronomy-aurora-forecast-space-weather"
    article_title, article_content = get_article_content(url)
    
    print(f"Заголовок статьи: {article_title}\n")
    print("Содержание статьи:")
    print(article_content)
