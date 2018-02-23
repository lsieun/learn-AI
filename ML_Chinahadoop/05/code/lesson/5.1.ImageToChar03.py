# coding:utf-8
# 知识点：（1）对图片进行缩放
from PIL import Image

if __name__ == '__main__':
    # (1) open image file
    img_path = '.\\images\\afen.png'
    img = Image.open(img_path)

    # (2) get a new image size
    old_img_width, old_img_height = img.size
    new_img_width = 2 * old_img_width
    new_img_height = 2 * old_img_height
    print('old_img_width = ', old_img_width, ', old_img_height = ', old_img_height)
    print('new_img_width = ', new_img_width, ', new_img_height = ', new_img_height)

    # (3) resize an image
    new_img = img.resize((new_img_width, new_img_height), Image.ANTIALIAS)

    # (4) save an image
    new_img.save('.\\tmp_imgs\\resize_img.png')
    print("Save Success!!!")