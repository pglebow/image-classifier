__author__ = 'pglebow'
from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)

image_type = "shirt"
query = "shirt"
url = "http://www.bing.com/images/search?q=" + query + \
    "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

soup = get_soup(url)
images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]

for img in images:
    raw_img = urllib2.urlopen(img).read()
    cntr = len([i for i in os.listdir("images") if image_type in i]) + 1
    f = open("images/" + image_type + "_"+ str(cntr) + ".jpg", 'wb')
    f.write(raw_img)
    f.close()