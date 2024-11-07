import requests
from bs4 import BeautifulSoup
import pandas as pd



def get_soup(url):
  response = requests.get(url)
  html_content = response.text
  soup = BeautifulSoup(html_content, 'html.parser')
  return soup

def get_constellations(url):
  constellations_soup = get_soup(url)
  tbody = constellations_soup.find('tbody')
  constellations = tbody.find_all('tr')
  return constellations

def get_constellations_description(url):
  constellations_info = {}
  constellations_page= get_soup(url)
  description_list = constellations_page.find('div', attrs={'class': "newsbody"}).findAll('p', attrs={'class': None})
  description = ''
  for p in description_list:
    description += p.text.strip().replace("\n", " ")
  constellations_info['description'] = description
  additional_info_block = constellations_page.findAll('div', attrs={'class': "objinfo"})
  additional_info_list = [info.find_all(string=True) for info in additional_info_block]
  additional_info_dict = {}
  for info in additional_info_list:
    additional_info_dict[info[0]] = " ".join(word.strip() for word in info[1:])
  constellations_info['additional_info'] = additional_info_dict
  constellation_image = constellations_page.find('img')['src']
  constellations_info['img_src'] = constellation_image
  return constellations_info

def get_brightest_star_info(url):
  brightest_star_info = {}
  brightest_star_page = get_soup(url)
  brightest_star_description = brightest_star_page.find('p', attrs={'class': "newstext"}).text.strip().replace("\n", " ")
  brightest_star_info['description'] = brightest_star_description
  return brightest_star_info


constellations = get_constellations('https://in-the-sky.org/data/constellations_list.php')

constellations_data = []
for constellation in constellations:
  tds = constellation.find_all('td')
  constellation_dict = {}
  constellation_dict['name'] = tds[0].text.strip()
  constellation_dict['short_description'] = tds[1].find('div').text.strip()
  constellation_dict['first_appeared'] = tds[2].find('div').text.strip()
  full_description_url = tds[0].find('a')['href']
  constellation_dict['full_description'] = get_constellations_description(full_description_url)
  constellation_dict['genitive_form'] = tds[3].text.strip()
  constellation_dict['brightest_star'] = tds[4].text.strip()
  star_info = get_brightest_star_info(tds[4].find('a')['href'])
  constellation_dict['brightest_star_info'] = star_info
  constellations_data.append(constellation_dict)


df = pd.DataFrame(constellations_data)
df.to_json('datasets/constellation_text_data/constellation_data.json', orient='records')