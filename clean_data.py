# Do the following things for each news in the dataset:
# 1. split into sentences
# 2. remove puncs and non-letter words
# 3. lowercase
# 4. stop word removal
# 5. stemming

import argparse
import csv
import json
import re

import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


parser = argparse.ArgumentParser(description='Clean csv news file and save as json')
parser.add_argument("--csv_dataset", type=str, default="data/processed/politifact_train.csv",
                    help="CSV dataset")
parser.add_argument("--output", type=str, default="data/processed/politifact_train.cleaned.json",
                    help="Output json file")

args = parser.parse_args()

nltk.download('punkt')
nltk.download('stopwords')

# Read from csv
rows = []
with open(args.csv_dataset) as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        rows.append(row)
print("Read %d rows from %s" % (len(rows), args.csv_dataset))

regex = re.compile('[^a-zA-Z ]+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for i in range(len(rows)):
    # split into sentences
    sentences = sent_tokenize(rows[i]["text"])
    # tokenization
    for j in range(len(sentences)):
        sentences[j] = word_tokenize(sentences[j])
    # remove puncs and non-letter words
    for j in range(len(sentences)):
        new_words = []
        for k in range(len(sentences[j])):
            word = regex.sub(' ', sentences[j][k])
            if len(word.strip()) > 0:
                new_words += word.strip().split()
        sentences[j] = new_words
    # lowercase
    for j in range(len(sentences)):
        for k in range(len(sentences[j])):
            sentences[j][k] = sentences[j][k].lower()
    # stop word removal
    for j in range(len(sentences)):
        new_words = []
        for word in sentences[j]:
            if word not in stop_words:
                new_words.append(word)
        sentences[j] = new_words
    # stemming
    for j in range(len(sentences)):
        for k in range(len(sentences[j])):
            sentences[j][k] = ps.stem(sentences[j][k])

    if i < 3:
        print("Data #%d" % i)
        print(rows[i])
        print(sentences)

    rows[i]["text"] = sentences

with open(args.output, "w") as f:
    json.dump(rows, f)
