import argparse
import json

from scipy.stats import pearsonr
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def calc_tfidf(dataset, ngram):
    # extract the n-gram features
    all_word_cnt = {}
    doc_word_cnt = []

    for d in dataset:
        word_cnt = {}
        for sentence in d["text"]:
            n = len(sentence)
            for i in range(n):
                if i + ngram < n:
                    word = tuple(sentence[i:i+ngram])
                    word_cnt[word] = word_cnt.get(word, 0) + 1
                    all_word_cnt[word] = all_word_cnt.get(word, 0) + 1
        doc_word_cnt.append(word_cnt)

    all_word_list = list(all_word_cnt.keys())
    all_word_list = sorted(all_word_list)
    print("Number of n-gram features: %d" % len(all_word_list))
    # print(all_word_list)

    n = len(dataset)
    m = len(all_word_list)
    X = np.zeros([n, m])
    for i in range(n):
        for j, word in enumerate(all_word_list):
            X[i][j] = doc_word_cnt[i].get(word, 0)
    # print(X)

    tfidf = TfidfTransformer()
    X_new = tfidf.fit_transform(X=X)
    # print(X_new)

    return all_word_list, np.asarray(X_new.todense())


def select_feature(all_word_list, X, feature_list):
    # select the features that are previously selected (0 if not exist)
    n, m = X.shape
    valid, invalid = 0, 0
    all_word_dict = {}
    for i, word in enumerate(all_word_list):
        all_word_dict[word] = i
    columns = []
    for word in feature_list:
        if word in all_word_dict:
            i = all_word_dict[word]
            columns.append(X[:, i])
            valid += 1
        else:
            columns.append(np.zeros([n], dtype=np.float))
            invalid += 1
    print("Found %f%% percent valid feature" % (valid / (valid + invalid) * 100) )
    return np.column_stack(columns)


parser = argparse.ArgumentParser(description='Extract features and train')
parser.add_argument("--train", type=str, default="data/processed/politifact_train.cleaned.json",
                    help="training dataset")
parser.add_argument("--test", type=str, default="data/processed/politifact_test.cleaned.json",
                    help="test dataset")
parser.add_argument("--ngram", type=int, default=1,
                    help="N-gram number")
parser.add_argument("--n_features", type=int, default=1000,
                    help="Number of features")
parser.add_argument("--classifier", type=str, default="SVM",
                    help="Type of classifier to use")

args = parser.parse_args()

with open(args.train, "r") as f:
    train_dataset = json.load(f)

with open(args.test, "r") as f:
    test_dataset = json.load(f)

# Calculate tf-idf
all_word_list_train, X_train = calc_tfidf(train_dataset, args.ngram)
y_train = [1 if row["label"] == "real" else 0 for row in train_dataset]
all_word_list_test, X_test = calc_tfidf(test_dataset, args.ngram)
y_test = [1 if row["label"] == "real" else 0 for row in test_dataset]
print("Finished calculating tf-idf")

# calculate pearsonr for each feature
# print(X)
n, m = X_train.shape
pearson = np.zeros([m])
for i in range(m):
    x = X_train[:, i]
    pearson[i] = pearsonr(x, y_train)[0]

# find out the top K features
print("Calculated top pearsonr values")
K = min(args.n_features, m)
idx = np.argpartition(-pearson, K)
# print(pearson[idx[:K]])
feature_list = []
for i in idx[:K]:
    feature_list.append(all_word_list_train[i])

# filter out for the top features
X_train_filtered = select_feature(all_word_list_train, X_train, feature_list)
X_test_filtered = select_feature(all_word_list_test, X_test, feature_list)
print(X_train_filtered.shape)
print(X_test_filtered.shape)

if args.classifier == "SVM":
    clf = svm.SVC()
    clf.fit(X_train_filtered, y_train)
    y_test_pred = clf.predict(X_test_filtered)
elif args.classifier == "KNN":
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train_filtered, y_train)
    y_test_pred = neigh.predict(X_test_filtered)
elif args.classifier == "DT":
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train_filtered, y_train)
    y_test_pred = clf.predict(X_test_filtered)
elif args.classifier == "LR":
    clf = LogisticRegression(random_state=0).fit(X_train_filtered, y_train)
    y_test_pred = clf.predict(X_test_filtered)
else:
    raise ValueError("Unknown classification method")

print("ACC: %f" % (y_test == y_test_pred).mean())
