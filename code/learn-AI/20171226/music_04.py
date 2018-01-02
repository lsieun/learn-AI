from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

# 对单首音乐进行傅里叶变换

plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
sample_rate, X = wavfile.read("d:/tmp/genres/jazz/converted/jazz.00000.au.wav")

plt.subplot(2,1,1)
specgram(X, Fs=sample_rate, xextent=(0,30))
plt.xlabel("time")
plt.ylabel("frequency")

plt.subplot(2,1,2)
fft_X = abs(fft(X), Fs=sample_rate, xextent=(0,30))
specgram(fft_X)
plt.xlabel("frequency")
plt.ylabel("amplitude")

plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
# plt.savefig("E:/genres/jazz.00000.au.wav.fft.png", bbox_inches="tight")
plt.show()