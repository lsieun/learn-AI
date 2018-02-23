# coding:utf-8
# 知识点：使用MD5和sha1加密

import hashlib

if __name__ == '__main__':
    md5 = hashlib.new('md5', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(md5.hexdigest())
    sha1 = hashlib.new('sha1', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(sha1.hexdigest())

    print(hashlib.algorithms_available)