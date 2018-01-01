from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

__author__ = 'yasaka'

# 可以先把一个wav文件读入python,然后绘制它的频谱图(spectrogram)来看看是什么样的

plt.figure(figsize=(10, 4), dpi=80)

# 采样率
(sample_rate, X) = wavfile.read("D:/tmp/genres/metal/converted/metal.00068.au.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.xlabel("time")
plt.ylabel("frequency")

plt.grid(True, linestyle='-', color='0.75')
# plt.savefig("D:/metal.00068.au.wav.png", bbox_inches="tight")
plt.show()