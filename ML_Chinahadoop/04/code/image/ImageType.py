import imghdr
import os

def calc_type():
    path = 'D:\\tmp\\images'
    cnt = 0
    for image_classes in os.listdir(path):
        image_classes_all = path + '\\' + image_classes
        for image_path in os.listdir(image_classes_all):
            image_path_all = image_classes_all + '\\' + image_path
            type = imghdr.what(image_path_all)
            if type != 'jpeg' and type != 'png':
                os.remove(image_path_all)
                print(image_path_all)
                cnt += 1
    print(cnt)

if __name__ == '__main__':
    calc_type()