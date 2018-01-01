#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: music.py

from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

__author__ = 'yasaka'

# 可以先把一个wav文件读入python,然后绘制它的频谱图(spectrogram)来看看是什么样的

plt.figure(figsize=(10, 4), dpi=80)

# 采样率
(sample_rate, X) = wavfile.read("E:/genres/metal/converted/metal.00068.au.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.xlabel("time")
plt.ylabel("frequency")

plt.grid(True, linestyle='-', color='0.75')
plt.savefig("D:/metal.00068.au.wav.png", bbox_inches="tight")

# 当然,我们也可以把每一种的音乐都抽一些出来打印频谱图以便比较,如下图:
def plotSpec(g,n):
    sample_rate, X = wavfile.read("E:/genres/"+g+"/converted/"+g+"."+n+".au.wav")
    specgram(X, Fs=sample_rate, xextent=(0,30))
    plt.title(g+"_"+n[-1])

# plt.figure(num=None, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(6,3,1);plotSpec("classical","00001");plt.subplot(6,3,2);plotSpec("classical","00002")
# plt.subplot(6,3,3);plotSpec("classical","00003");plt.subplot(6,3,4);plotSpec("jazz","00001")
# plt.subplot(6,3,5);plotSpec("jazz","00002");plt.subplot(6,3,6);plotSpec("jazz","00003")
# plt.subplot(6,3,7);plotSpec("country","00001");plt.subplot(6,3,8);plotSpec("country","00002")
# plt.subplot(6,3,9);plotSpec("country","00003");plt.subplot(6,3,10);plotSpec("pop","00001")
# plt.subplot(6,3,11);plotSpec("pop","00002");plt.subplot(6,3,12);plotSpec("pop","00003")
# plt.subplot(6,3,13);plotSpec("rock","00001");plt.subplot(6,3,14);plotSpec("rock","00002")
# plt.subplot(6,3,15);plotSpec("rock","00003");plt.subplot(6,3,16);plotSpec("metal","00001")
# plt.subplot(6,3,17);plotSpec("metal","00002");plt.subplot(6,3,18);plotSpec("metal","00003")
# plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
# plt.savefig("D:/compare.au.wav.png", bbox_inches="tight")


# 快速傅里叶变换
# FFT是一种数据处理技巧,它可以把time domain上的数据,例如一个音频,拆成一堆基准频率,然后投射到frequency domain上
# 为了理解FFT,我们可以先生成三个音频文件
"""
C:\sox-14-4-2\sox.exe --null -r 22050 d:\sine_a.wav synth 0.2 sine 400
C:\sox-14-4-2\sox.exe --null -r 22050 d:\sine_b.wav synth 0.2 sine 3000
C:\sox-14-4-2\sox.exe --combine mix --volume 1 sine_b.wav --volume 0.5 sine_a.wav sine_mix.wav
"""

# plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(3,2,1)
# sample_rate, a = wavfile.read("d:/sine_a.wav")
# specgram(a, Fs=sample_rate, xextent=(0,30))
# plt.xlabel("time")
# plt.ylabel("frequency")
# plt.title("400 HZ sine wave")
# plt.subplot(3,2,2)
# fft_a = abs(fft(a))
# specgram(fft_a)
# plt.xlabel("frequency")
# plt.ylabel("amplitude")
# plt.title("FFT of 400 HZ sine wave")
# plt.subplot(3,2,3)
# sample_rate, b = wavfile.read("d:/sine_b.wav")
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
# sample_rate, c = wavfile.read("d:/sine_mix.wav")
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


# 对单首音乐进行傅里叶变换

# plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
# sample_rate, X = wavfile.read("E:/genres/jazz/converted/jazz.00000.au.wav")
# plt.subplot(2,1,1)
# specgram(X, Fs=sample_rate, xextent=(0,30))
# plt.xlabel("time")
# plt.ylabel("frequency")
# plt.subplot(2,1,2)
# fft_X = abs(fft(X))
# specgram(fft_X)
# plt.xlabel("frequency")
# plt.ylabel("amplitude")
# plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
# plt.savefig("E:/genres/jazz.00000.au.wav.fft.png", bbox_inches="tight")
