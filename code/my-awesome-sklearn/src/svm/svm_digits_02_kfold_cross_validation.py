from sklearn import datasets,svm

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# To get a better measure of prediction accuracy, 
# we can successively split the data in folds 
# that we use for training and testing:

import numpy as np

X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
svc = svm.SVC(C=1,kernel='linear')

for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
    
print(scores)