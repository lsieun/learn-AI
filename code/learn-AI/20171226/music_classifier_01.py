import numpy as np
from scipy import fft
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
import pickle
import pprint

# 准备音乐数据

def create_fft(g, n):
    rad = "d:/tmp/genres/" + g + "/converted/" + g + "." + str(n).zfill(5) + ".au.wav"
    (sample_rate, X) = wavfile.read(rad)
    fft_features = abs(fft(X)[:1000])
    sad = "d:/tmp/trainset/" + g + "." + str(n).zfill(5) + ".fft"
    np.save(sad, fft_features)


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
for g in genre_list:
    for n in range(100):
        create_fft(g, n)

