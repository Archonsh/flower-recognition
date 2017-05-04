from bs4 import BeautifulSoup
import requests

url = "http://www.proflowers.com/blog/types-of-flowers"

headers = {
    'cache-control': "no-cache",
    'user-agent': 'Mozilla/5.0'
}

re = requests.request("GET", url, headers=headers)

soup = BeautifulSoup(re.text, 'html.parser')

h3s = soup.find_all("h3", class_="flower_name")

flowers = open('flowers.txt','wb')

for h3 in h3s:
    flowers.write(h3.text + '\n')
