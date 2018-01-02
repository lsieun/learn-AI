from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

sample_rate, a = wavfile.read("d:/tmp/sine_a.wav")

# plt.subplot(3,2,1)
# specgram(a, Fs=sample_rate, xextent=(0,30))
# plt.xlabel("time")
# plt.ylabel("frequency")
# plt.title("400 HZ sine wave")

plt.subplot(3,2,2)
fft_a = abs(fft(a))
specgram(fft_a)
# plt.xlabel("frequency")
# plt.ylabel("amplitude")
# plt.title("FFT of 400 HZ sine wave")

# plt.subplot(3,2,3)
# sample_rate, b = wavfile.read("d:/tmp/sine_b.wav")
# specgram(b, Fs=sample_rate, xextent=(0,30))
# plt.xlabel("time")
# plt.ylabel("frequency")
# plt.title("3000 HZ sine wave")
# plt.subplot(3,2,4)
# fft_b = abs(fft(b))
# specgram(fft_b)
# plt.xlabel("frequency")
# plt.ylabel("amplitude")
# plt.title("FFT of 3000 HZ sine wave")

# plt.subplot(3,2,5)
# sample_rate, c = wavfile.read("d:/tmp/sine_mix.wav")
# specgram(c, Fs=sample_rate, xextent=(0,30))
# plt.xlabel("time")
# plt.ylabel("frequency")
# plt.title("Mixed sine wave")
# plt.subplot(3,2,6)
# fft_c = abs(fft(c))
# specgram(fft_c)
# plt.xlabel("frequency")
# plt.ylabel("amplitude")
# plt.title("FFT of mixed sine wave")
# plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
# plt.savefig("D:/compare.sina.wave.png", bbox_inches="tight")

plt.show()