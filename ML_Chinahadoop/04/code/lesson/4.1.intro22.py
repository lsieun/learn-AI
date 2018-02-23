import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# Bar
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.bar(x, y, width=0.04, linewidth=0.2)
plt.plot(x, y, 'r--', linewidth=2)
plt.title(u'Sin曲线')
plt.xticks(rotation=-60)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()