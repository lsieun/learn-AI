import numpy as np
from scipy import fft
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
import pickle
import pprint

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]

pkl_file = open('data.pkl', 'rb')
model_load = pickle.load(pkl_file)
pprint.pprint(model_load)
pkl_file.close()

print('Starting read wavfile...')
sample_rate, test = wavfile.read("d:/tmp/sample/heibao-wudizirong-remix.wav")
testdata_fft_features = abs(fft(test))[:1000]

# print(sample_rate);
# print("-"*40)
# print(testdata_fft_features);
# print("-"*40)
# print(len(testdata_fft_features));

# print(sample_rate, testdata_fft_features, len(testdata_fft_features))


type_index = model_load.predict([testdata_fft_features])[0]
print(model_load.predict([testdata_fft_features]))
print(model_load.predict_proba([testdata_fft_features]))  #打印的是各个概率
print(type_index)
print(genre_list[type_index])