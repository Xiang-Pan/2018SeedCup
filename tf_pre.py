import nltk
import math
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
from nltk.stem.porter import *

print("hello")
all_w = []
title_w = []
dsc_w = []

def getData():
    data=pd.read_table('train_a_c.txt')
    data['combined'] = data['title_words']+data['description_words']
    print(data)
    return data


if __name__ == '__main__':
    data=getData()
    feature_extraction = TfidfVectorizer()
    X_train = feature_extraction.fit_transform(data['combined'].values)
    Y_train = data['cate3_id'].values
    print(X_train)
