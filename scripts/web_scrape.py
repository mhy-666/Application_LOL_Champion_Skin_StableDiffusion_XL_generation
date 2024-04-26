import requests
from bs4 import BeautifulSoup

class ChampionImageDownloader:
    def __init__(self, champion_file):
        self.champion_urls = self.get_champion_urls(champion_file)

    def get_champion_urls(self, champion_file):
        champion = []
        with open(champion_file, 'r') as f:
            for line in f:
                cleaned_line = ''.join(e for e in line.strip() if e.isalnum())
                champion.append(cleaned_line.lower())
        champion_urls = []
        for c in champion:
            if c == 'renataglasc':
                c = 'renata'
            champion_urls.append(f'https://www.skinexplorer.lol/champions/{c}')
        return champion_urls

    def get_image_url(self, page_url):
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        download_link = soup.find('main', style="transform:translateX(0)").find('img')['src']
        return download_link

    def download_image(self, image_url, save_path):
        response = requests.get(image_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)

    def get_thumbnail_links(self, output_dir):
        for url in self.champion_urls:
            response = requests.get(url)
            webpage = response.text
            champion_name = url.split('/')[-1]
            print(champion_name)
            soup = BeautifulSoup(webpage, 'html.parser')
            thumbnails = soup.find_all('div', class_="styles_grid__4dc4K")[0].find_all('a')
            thumbnail_urls = [f'https://www.skinexplorer.lol{thumb["href"]}' for thumb in thumbnails]
            full_image_urls = [self.get_image_url(url) for url in thumbnail_urls]
            for i, img_url in enumerate(full_image_urls):
                save_path = f'{output_dir}/{champion_name}_{i}.jpg'
                self.download_image(img_url, save_path)


class ChampionURLScraper:
    def __init__(self):
        self.champion_urls = []

    def get_champion_urls(self):
        champion = []
        with open('../data/raw/champion.txt', 'r') as f:
            for line in f:
                cleaned_line = ''.join(e for e in line.strip() if e.isalnum())
                champion.append(cleaned_line.lower())

        for c in champion:
            c = c.replace(" ", "").replace("'", "").replace("'", "")
            if c == 'renataglasc':
                c = 'renata'

            self.champion_urls.append('https://universe.leagueoflegends.com/en_US/story/champion/' + c)

    def scrape_champion_page(self, url):
        response = requests.get(url)
        webpage = response.text
        champion_name = url.split('/')[-1]
        print(champion_name)

        # Parse the webpage
        soup = BeautifulSoup(webpage, 'html.parser')

        # Find the thumbnail links
        # Note: 'thumbnail_class' should be replaced with the actual HTML class name of the thumbnail links
        thumbnails = soup.find_all('div', class_="root_3Kft")
        print(thumbnails)

        target_element = soup.select_one('CatchElement')
        print(target_element)

    

if __name__ == "__main__":
    downloader = ChampionImageDownloader('../data/raw/champion.txt')
    downloader.get_thumbnail_links('../data/raw/images')

    scraper = ChampionURLScraper()
    scraper.get_champion_urls()

    # Scrape the first champion page
    scraper.scrape_champion_page(scraper.champion_urls[0])