import numpy as np
from scipy import fft
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
import pickle
import pprint

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
X = []
Y = []
for g in genre_list:
    for n in range(100):
        rad = "d:/tmp/trainset/" + g + "." + str(n).zfill(5) + ".fft" + ".npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        Y.append(genre_list.index(g))

X = np.array(X)
Y = np.array(Y)

# print(X)
# print("-"*40)
# print(Y)

# 接下来，我们使用sklearn，来构造和训练我们的分类器
# ------train logistic classifier--------------

model = LogisticRegression()
model.fit(X, Y)

# 可以采用Python内建的持久性模型 pickle 来保存scikit的模型

output = open('data.pkl', 'wb')
pickle.dump(model, output)
output.close()
