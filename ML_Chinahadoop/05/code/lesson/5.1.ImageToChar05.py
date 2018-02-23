# coding:utf-8
# 知识点：（1）将图片的像素转换成字符

import numpy as np
from PIL import Image

if __name__ == '__main__':
    img = Image.open('.\\images\\afen.png')
    old_img_width, old_img_height = img.size

    new_img_height = 100
    new_img_width = int(1.5 * new_img_height * old_img_height // old_img_height)    # 假定字符的高度是宽度的1.5倍
    # print('old_img_width = ', old_img_width, ', old_img_height = ', old_img_height)
    # print('new_img_width = ', new_img_width, ', new_img_height = ', new_img_height)

    new_img = img.resize((new_img_width, new_img_height), Image.ANTIALIAS)
    pixels = np.array(new_img.convert('L'))
    row, col = pixels.shape

    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)
    step = 256 // N
    print(N)
    result = ''
    for i in range(row):
        for j in range(col):
            result += chars[pixels[i][j] // step]
        result += '\n'

    with open('.\\tmp_imgs\\acsii_img.txt', mode='w') as f:
        f.write(result)