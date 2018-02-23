# coding:utf-8
# 知识点：使用sha1加密

import hashlib

if __name__ == '__main__':
    sha1 = hashlib.sha1()
    sha1.update('Hello World.世界，你好!'.encode('utf-8'))
    print('SHA1:', sha1.hexdigest())
    print(sha1.digest_size, sha1.block_size)
    print('=====================')