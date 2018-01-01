from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

__author__ = 'yasaka'

# Alternative method to load MNIST, if mldata.org is down
from scipy.io import loadmat

mnist_raw = loadmat("./mldata/mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
print("Success!")
# mnist = fetch_mldata('MNIST_original', data_home='test_data_home')
# print(mnist)

X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)

some_digit = X[36000]
print(some_digit)
some_digit_image = some_digit.reshape(28, 28)
print(some_digit_image)

'''
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  86 131 225 225 225   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  13  73 197 253 252 252 252 252   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   4  29  29 154 187 252 252 253 252 252 233 145   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  29 252 253 252 252 252 252 253 204 112  37   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0 169 253 255 253 228 126   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  98 243 252 253 252 246 130  38   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  98 240 252 252 253 252 252 252 221   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 225 252 252 236 225 223 230 252 252   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 146 252 157  50   0   0  25 205 252   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  26 207 253   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0  29  19   0   0   0   0   0   0   0   0   0  73 205 252  79   0   0   0   0   0   0   0   0]
 [  0   0   0   0 120 215 209 175   0   0   0   0   0   0   0  19 209 252 220  79   0   0   0   0   0   0   0   0]
 [  0   0   0   0 174 252 252 239 140   0   0   0   0   0  29 104 252 249 177   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 174 252 252 223   0   0   0   0   0   0 174 252 252 223   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 141 241 253 146   0   0   0   0 169 253 255 253 253  84   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0 178 252 154  85  85 210 225 243 252 215 121  27   9   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0  66 208 220 252 253 252 252 214 195  31   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  19  37  84 146 223 114  28   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
'''

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()