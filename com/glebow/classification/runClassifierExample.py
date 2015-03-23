__author__ = 'pglebow'
from bs4 import BeautifulSoup
import requests
import re
import urllib2
import pandas as pd
import numpy as np
import pylab as pl
import glob as glob
from PIL import Image
import os
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)

def get_soup(url):
    return BeautifulSoup(requests.get(url).text)

    image_type = "shoe"
    query = "shoe"
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
    return 0


def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)

    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def processImages():
    img_dir = "images/"
    glob.glob('*.jpg')
    images = [img_dir+ f for f in os.listdir(img_dir) if any([f.endswith("jpg")])]
    labels = ["shoe" if "shoe" in f.split('/')[-1] else "shirt" for f in images]
    retVal = []
    localData = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        localData.append(img)

    retVal = np.array(localData)

    return retVal

def pca(imageData=[]):
    labels = ["shoe", "shirt"]
    is_train = np.random.uniform(0, 1, len(imageData)) <= 0.7
    y = np.where(np.array(labels) == "shirt", 1, 0)

    train_x, train_y = imageData[is_train], imageData[is_train]
    test_x, test_y = imageData[is_train==False], y[is_train==False]
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(imageData)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": np.where(y == 1, "shoe", "shirt")})
    colors = ["red", "yellow"]
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label']==label
        pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    pl.legend()
    pl.show()

    pca2 = RandomizedPCA(n_components=5)
    train_x = pca2.fit_transform(train_x)
    test_x = pca2.transform(test_x)

    print train_x[:5]
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)
    return 0

def main():
    p = processImages()
    pca(p)


if __name__ == "__main__":
    main()