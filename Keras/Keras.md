# Keras #

Keras官网: [https://keras.io/](https://keras.io/)

## Keras简介 ##

> 问题：你对Keras有了解吗？

Keras: The Python Deep Learning library

Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Keras is compatible with: Python 2.7-3.6.

公式总结：

- （1）Keras = Deep Learning Library + Python
- （2）Keras = High-level NN API(tensorflow/mirsoft cntk/theano)
- （3）Keras --> idea --> result --> fast experimentaion

文字总结：

- （1）Keras是一个deep learning library，是用Python语言开发的。（What:deep learning lib + python）
- （2）Keras属于high-level的神经网络API，能够在tensorflow、CNTK和Theano上面运行。（与其他神经网络框架的关系）
- （3）开发Keras的目的就是为了实现idea与result之间的Fast experimentation。

## 如何安装Keras ##

使用pip命令安装keras

	pip install --upgrade keras

	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade keras

## 打印Keras的版本 ##

代码

```python
import keras
print(keras.__version__)
```

输出：

	Using TensorFlow backend.
	2.1.5

## Keras的两种模型 ##

Keras的两种模型：

- （1）序列模型
- （2）通用模式

## Keras的序列模型 ##

**序列模型**属于**通用模型**的一种，因为很常见，所以这里单独列出来进行介绍。

序列模型，各层之间是依次顺序的线性关系。（序列模型-->各层-->线性关系）

序列模型的编程模型。在第k层和第k+1层之间可以加上**各种元素**来构造神经网络，**这些元素**可以通过一个**列表**来制定，然后作为**参数**传递给**序列模型**来生成相应的模型。

	脉络：各层-->各种元素-->列表-->参数-->序列模型

示例代码：

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

# Dense相当于构建一个全连接层，32指的是全连接层上面神经元的个数
layers = [Dense(units=32, input_shape=(784,)),
          Activation(activation='relu'),
          Dense(units=10),
          Activation(activation='softmax')]
model = Sequential(layers=layers)
model.summary()
```

输出：

	Using TensorFlow backend.
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	dense_1 (Dense)              (None, 32)                25120     
	_________________________________________________________________
	activation_1 (Activation)    (None, 32)                0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 10)                330       
	_________________________________________________________________
	activation_2 (Activation)    (None, 10)                0         
	=================================================================
	Total params: 25,450
	Trainable params: 25,450
	Non-trainable params: 0
	_________________________________________________________________

代码说明：

- （1）`Sequential`： Linear stack of layers.
- （2）`Dense`： Just your regular densely-connected NN layer. densely的意思是“浓密地，稠密地，密集地”，densely-connected NN layer应该理解为“全连接的神经网络”。
- （3）`Activation`： Applies an activation function to an output.
- （4）`model.summary()`： Prints a string summary of the network.

输出说明：

- （1）`dense_1 (Dense)`有25120个Param，本层一共有32个神经元节点，每个神经元与输入层的784个输入节点连接，再加上32个偏置项，一共是25120（32x784+32=25120）个连接参数。
- （2）`Total params`是25450（25450=25120+330）个。

## Keras的序列模型的另一种写法 ##

示例代码：

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

model = Sequential()
model.add(Dense(units=32, input_shape=(784,)))
model.add(Activation(activation='relu'))
model.add(Dense(units=10))
model.add(Activation(activation='softmax'))
model.summary(line_length=60)
```

输出：

	Using TensorFlow backend.
	____________________________________________________________
	Layer (type)               Output Shape            Param #  
	============================================================
	dense_1 (Dense)            (None, 32)              25120    
	____________________________________________________________
	activation_1 (Activation)  (None, 32)              0        
	____________________________________________________________
	dense_2 (Dense)            (None, 10)              330      
	____________________________________________________________
	activation_2 (Activation)  (None, 10)              0        
	============================================================
	Total params: 25,450
	Trainable params: 25,450
	Non-trainable params: 0
	____________________________________________________________

代码说明：

- （1）`model.summary(line_length=60)`中的`line_length`: Total length of printed lines

## Keras通用模型 ##

**通用模型**可以用来设计非常复杂、任意拓扑结构的神经网络，例如有向无环图网络。

类似于**序列模型**，**通用模型**通过**函数化的应用接口**来定义模型。使用函数化的应用接口有**好多好处**，比如：决定函数**执行结果**的唯一要素是**其返回值**，而决定**返回值**的唯一要素则是**其参数**，这大大减轻了代码测试的工作量

	脉络（1）：通用模型-->函数化的应用接口-->好多好处
	脉络（2）：好处-->函数-->执行结果-->返回值-->其参数

通用模型的**编程模型**。在通用模型中，定义的时候，从**输入**的多维矩阵开始，然后定义**各层及其要素**，最后定义**输出层**，将**输入层**和**输出层**作为**参数**纳入**通用模型**中就可以定义一个模型对象。

	脉络：输入（多维矩阵）-->各层要素-->输出层-->(输入层,输出层）-->通用模型

思路文字说明：假设从输入层开始；然后定义两个隐层，每个隐层有64个神经元，都使用relu激活函数；接着，输出层有10个神经元节点，使用softmax作为激活函数；最后，当所有要素都齐备以后，就可以定义**模型对象**了，参数很简单，分别是输入和输出，其中包含了中间的各种信息。当模型对象定义完成之后，就可以进行编译了，并对数据进行拟合，拟合的时候也有两个参数，分别对应于输入和输出。

示例代码：

```python
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

# 定义输入层
input = Input(shape=(784,))
# 定义各个连接层
x = Dense(units=64, activation='relu')(input)
x = Dense(units=64, activation='relu')(x)
# 定义输出层
y = Dense(units=10, activation='softmax')(x)

# 定义模型对象
model = Model(inputs=x, outputs=y)

# 模型编译和拟合
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=data, y=labels) # 注意：这里没有定义data和labels
```

代码说明：

- （1）Input： `Input()` is used to instantiate a Keras tensor.
- （2）`model.compile`的`optimizer`参数: String (name of optimizer) or optimizer instance.
- （3）`model.compile`的`loss`参数: String (name of objective function) or objective function. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.
- （4）`model.compile`的`metrics`参数: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as `metrics={'output_a': 'accuracy'}`.

## 使用Keras训练MNIST数据 ##





















