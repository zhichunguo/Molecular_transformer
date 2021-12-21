from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
from scipy.special import softmax
import numpy as np
xx = clf.predict_proba(X)
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0) # only difference

print(xx)
print(softmax(xx, axis=1))


print(clf.predict_proba(X)[:, 1].shape)
print(roc_auc_score(y, softmax(xx, axis=1)[:, 0]))
print(clf.decision_function(X).shape)
print(roc_auc_score(y, softmax(xx, axis=1)[:, 1]))
raise TypeError


from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(solver="liblinear").fit(X, y)

# print((y-1)**2)
# raise TypeError
# print(clf.predict_proba(X)[:, :2].shape)
print(roc_auc_score((y-1)**2, clf.predict_proba(X)[:, :2]))


# print(type(y))

# print(type(clf.predict_proba(X)))

# roc_auc_score(y, clf.predict_proba(X)[:, 1])
# 0.99...
# >>> roc_auc_score(y, clf.decision_function(X))
# 0.99...