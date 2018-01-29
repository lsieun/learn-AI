from sklearn import datasets,svm

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1,kernel='linear')
model = svc.fit(X_digits[:-100],y_digits[:-100])
# Every estimator exposes a 'score' method that 
# can judge the quality of the 'fit' (or the prediction) on new data. 
# Bigger is better.
score = model.score(X_digits[-100:],y_digits[-100:])
print(score)