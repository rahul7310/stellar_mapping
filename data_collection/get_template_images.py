import urllib.request
from  bs4 import BeautifulSoup
import requests
import ssl


#Get constellation names list
url = 'https://starchild.gsfc.nasa.gov/docs/StarChild/questions/88constellations.html'
response = requests.get(url)
html_content = response.text
soup_constellations_list = BeautifulSoup(html_content, 'html.parser')
tr_elements = soup_constellations_list.find_all('tr')

constellations_list = [tr_element.find('td').text.strip().replace(' ', '') for tr_element in tr_elements[1:-1]]
#Replace the ones with different names
constellations_list[1] = 'Antilia'
constellations_list[53] = 'Microscopus'
constellations_list[80] = 'TriangulumAustralis'


#Get HTML content for constellation images
URL = f"https://astronomyonline.org/Observation/Constellations.asp?Cate=Observation&SubCate=MP07&SubCate2=MP0801"
html_page = requests.get(URL)
soup_constellation_images = BeautifulSoup(html_page.content, 'html.parser')


#Function to extract the image urls
def extract_image_url(constellation_name, soup):
   a_tags = soup.find_all('a', href=lambda value: constellation_name in value if value else False)
   if not a_tags:
       print(f"No image found for {constellation_name}")
       return None
   return a_tags[0].find('img')['src'].replace('Small', "Big")

#Function to download the images
def download_image(image_url, save_path, constellation_name):
    
    full_img_url = "https://astronomyonline.org/Observation/" + image_url

    # print(f"Downloading: {full_img_url}")
    try:
        context = ssl._create_unverified_context()
        # Use urlopen with the context
        with urllib.request.urlopen(full_img_url, context=context) as response:
            img_data = response.read()
            
        # Save the image data
        with open(save_path, 'wb') as f:
            f.write(img_data)
        # print("Downloaded successfully.")
    except Exception as e:
        print(f"Error downloading {constellation_name}: {e}")


#Loop through constellation names and download the image for each constellation
for constellation in constellations_list:
  print(constellation)
  img_url = extract_image_url(constellation, soup_constellation_images)
  download_image(img_url, f"datasets/constellation_templates/{constellation}.jpg", constellation)

