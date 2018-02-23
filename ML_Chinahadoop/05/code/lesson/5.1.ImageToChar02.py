# coding:utf-8
# 知识点：（1）打开图片文件；（2）显示图片的长宽大小
from PIL import Image



if __name__ == '__main__':
    # (1) open image file
    img_path = '.\\images\\afen.png'
    img = Image.open(img_path)  # img_path的文件路径必须真实存在；img是一个PIL.PngImagePlugin.PngImageFile对象
    print('img = ', img)        # img =  <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=207x206 at 0x28E5D68>
    print(type(img))            # <class 'PIL.PngImagePlugin.PngImageFile'>
    print("="*40)

    # (2) display image size
    img_size = img.size         # 图片长和宽，单位是像素
    print(img.size)             # (207, 206)
    print(type(img_size))       # 数据类型是：<class 'tuple'>
    print("=" * 40)
