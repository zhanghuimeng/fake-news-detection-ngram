# Parse data from downloaded FakeNewsNet

import argparse
import csv
import json
import os
import random

parser = argparse.ArgumentParser(description='Parse news to csv file')
parser.add_argument("--small_data", type=str, default="data/fakenewsnet_small",
                    help="Small data to discard")
parser.add_argument("--large_data", type=str, default="data/fakenewsnet_dataset",
                    help="Large data to parse")
parser.add_argument("--output", type=str, default="data/processed",
                    help="Output directory")

args = parser.parse_args()

# get all the titles in small dataset
titleSet = set()
files = ["BuzzFeed_fake_news_content.csv", "BuzzFeed_real_news_content.csv",
         "PolitiFact_fake_news_content.csv", "PolitiFact_real_news_content.csv"]
for filename in files:
    with open(os.path.join(args.small_data, filename)) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # print(row["title"])
            titleSet.add(row["title"])
print("Found %d news in small dataset" % len(titleSet))

# read big dataset
subdirs = ["gossipcop/fake", "gossipcop/real", "politifact/fake", "politifact/real"]
source = ["gossipcop", "gossipcop", "politifact", "politifact"]
label = ["fake", "real", "fake", "real"]

headers = ["source", "label", "title", "text"]
g_rows = []
p_rows = []

for i, subdir in enumerate(subdirs):
    content_dirs = [f.path for f in os.scandir(os.path.join(args.large_data, subdir)) if f.is_dir()]
    for c_dir in content_dirs:
        # print(c_dir)
        filepath = os.path.join(c_dir, "news content.json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                obj = json.load(f)
            if obj["title"] in titleSet:
                continue
            row = {"source": source[i], "label": label[i]}
            row["title"] = obj["title"]
            row["text"] = obj["text"]

            if source[i] == "politifact":
                p_rows.append(row)
            else:
                g_rows.append(row)

print("Found %d gossipcop news" % len(g_rows))
print("Found %d politifact news" % len(p_rows))


def shuffle_split(data, p=0.2):
    l = int((1 - p) * len(data))
    random.shuffle(data)
    return data[:l], data[l:]


# output
with open(os.path.join(args.output, "gossipcop_all.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(g_rows)
    print("Printed %d gossipcop news" % len(g_rows))
with open(os.path.join(args.output, "politifact_all.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(p_rows)
    print("Printed %d politifact news" % len(p_rows))
exit()

g_train, g_test = shuffle_split(g_rows)
with open(os.path.join(args.output, "gossipcop_train.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(g_train)
    print("Printed %d training set of gossipcop news" % len(g_train))
with open(os.path.join(args.output, "gossipcop_test.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(g_test)
    print("Printed %d test set of gossipcop news" % len(g_test))

p_train, p_test = shuffle_split(p_rows)
with open(os.path.join(args.output, "politifact_train.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(p_train)
    print("Printed %d training set of politifact news" % len(p_train))
with open(os.path.join(args.output, "politifact_test.csv"), "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(p_test)
    print("Printed %d test set of politifact news" % len(p_test))
