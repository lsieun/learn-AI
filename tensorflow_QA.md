# Tensorflow常见问题 #

## 第01个问题 ##


出现如下提示：

	The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.

只需要在相应的.py文件头这样写：

```python
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  
import tensorflow as tf  
  
...  
```

然后就没有问题啦～

```python
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # 这是默认的显示等级，显示所有信息  
  
# 2级  
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 只显示 warning 和 Error  
  
# 3级  
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 只显示 Error 
```

