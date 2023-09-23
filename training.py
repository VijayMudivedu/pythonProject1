from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Dataframe
X, y = make_classification(n_samples=1000, random_state=12, n_classes=2, n_features=20)
X, y  # Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)

[i.shape for i in [X_train, X_test, y_train, y_test]]

reg_model = LogisticRegression()
knn_model = KNeighborsClassifier(n_neighbors=4)

reg_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

pred_prob1 = reg_model.predict_proba(X_test)
pred_prob2 = knn_model.predict_proba(X_test)
pred_prob1.shape

fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1])
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1])

# Random Probability
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.plot(fpr1, tpr1, ls="-", color='r', label="Logistic")
plt.plot(fpr2, tpr2, ls="-.", c='b', label='knn')
plt.plot(p_fpr, p_tpr, c='g', ls='--', label='random')

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="best")
plt.show()

roc_auc_score(y_test, pred_prob1[:, 1], max_fpr=0)


#####################
from sklearn.multiclass import OneVsRestClassifier

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=3, random_state=42)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# fit model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)

# roc curve for classes
fpr = {}
tpr = {}
thresh = {}

n_class = 3


for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

# plotting
color = ['r','b','g']
for i in range(n_class):
    plt.plot(fpr[i], tpr[i], linestyle='--', color=color[i], label=f'Class {i} vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')

plt.legend(loc="best")

X[1].mean()
X[1].sum()

import numpy as np
np.median(X[1])

import statistics as st
n = 5
x = [95, 85, 80, 70, 60]
y = [85, 95, 70, 65, 70]

mean_x = st.mean(x)
mean_y = st.mean(y)

x_sq = sum( map(lambda x: pow(x,2), x))
xy = sum(map(lambda a,b: a*b, x,y))

sum([x[i]**2 for i in range(n)])

# Set the B and A
b = (n * xy - sum(x) * sum(y)) / (n * x_sq - (sum(x) ** 2))
a = mean_y - b * mean_x

# Gets the result and show on the screen
print(round(a + 80 * b, 3))

def fact(n):
    return 0 if n==0 else 1 if n==1 else fact(n-1)+fact(n-2)

fact(11)

0,1,1,2,3,5,8,13,21,34,55,89