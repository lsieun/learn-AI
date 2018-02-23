#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

if __name__ == '__main__':
    image_file = 'afen.png'
    height = 100

    img = Image.open(image_file)
    #print('img = ',img) # img =  <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=207x206 at 0x291C240>
    img_width, img_height = img.size
    print('img.size = ',img.size)
    width = 2 * height * img_width // img_height    # 假定字符的高度是宽度的2倍
    img = img.resize((width, height), Image.ANTIALIAS)
    print('img.size = ', img.size)
    pixels = np.array(img.convert('L'))
    print(pixels.shape)
    print(pixels)
    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)
    step = 256 // N
    print(N)
    result = ''
    for i in range(height):
        for j in range(width):
            result += chars[pixels[i][j] // step]
        result += '\n'
    with open('text.txt', mode='w') as f:
        f.write(result)
