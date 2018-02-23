# coding:utf-8
# 知识点：使用MD5加密

import hashlib

if __name__ == '__main__':
    md5 = hashlib.md5()
    md5.update('Hello World.世界，你好!'.encode('utf-8'))
    print('MD5:', md5.hexdigest())
    print(md5.digest_size, md5.block_size)
    print('------------------')