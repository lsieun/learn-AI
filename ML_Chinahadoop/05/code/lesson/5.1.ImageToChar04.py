# coding:utf-8
# 知识点：（1）将彩色图片转换成灰度图片
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # (1) open image file
    img_path = '.\\images\\afen.png'
    img = Image.open(img_path)

    # (2) 生成灰度图
    convert_img = img.convert('L')
    print(convert_img)         # <PIL.Image.Image image mode=L size=207x206 at 0x2934128>
    print(type(convert_img))   # <class 'PIL.Image.Image'>

    # (3) save an image
    convert_img.save('.\\tmp_imgs\\Gray_Scale_Image.png')

    # (4) 转换成numpy.ndarray对象
    pixels = np.array(img.convert('L'))
    print(type(pixels))
    print('pixels.shape = ', pixels.shape)
    print(pixels)