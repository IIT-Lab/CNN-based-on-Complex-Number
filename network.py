import tensorflow as tf
import numpy as np

# 复数张量的表示：原通道数为n，则将通道数分为2n，前n表示实部，后n表示虚部。
IMAGE_SIZE=28
IMAGE_CHANNEL=2  #实部+虚部
FILTER1_SIZE=3
FILTER1_NUM=64
FILTER2_SIZE=3
FILTER2_NUM=64
FILTER3_SIZE=3
FILTER3_NUM=64
FC1_SIZE=128       # 指的是有128个复数
FC2_SIZE=128
OUTPUT_NODE=10


def get_weight(shape,regularizer):
    # shape: 同卷积核维度 [行，列，通道，个数]
    w=tf.Variable(tf.truncated_normal(shape,stddev=1))
    if regularizer!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    # shape:卷积核个数 [FILTER_NUM]
    return tf.Variable(tf.zeros(shape))


def complex_conv(input,filter,strides):
    # input: [batch,行，列，通道]    前一半通道是实数，后一半是复数
    # filter: [行，列，通道，滤波器个数]
    # strides: 核滑动步长[1,行步长，列步长，1]

    # W=A+Bi
    # f=x+yi
    [A,B]=tf.split(input,2,3)   # 在第3维（通道）上切割  （从第0维开始计数）
    [x,y]=tf.split(filter,2,2)  # 在第2维（通道）上切割
    output1= tf.nn.conv2d(A,x,strides,'SAME') - tf.nn.conv2d(B,y,strides,'SAME') #实部
    output2= tf.nn.conv2d(A,y,strides,'SAME') + tf.nn.conv2d(B,x,strides,'SAME') #虚部
    output=tf.concat([output1,output2],3)   # 在通道上拼接

    return output

def modRelu(input):
    # input: [batch，行，列，通道]
    A, B = tf.split(input, 2, 3)  # 在第3维（通道）上切割
    C = tf.abs(tf.complex(A, B))

    # 可以学习的变量b
    b = 50 * np.ones([C.get_shape()[-1]])
    b = tf.Variable(b, dtype=tf.float32)
    C1=C-b
    # 衰减系数fac
    fac = C1/(C+0.0001)

    relu=tf.nn.relu(C1)
    flag=tf.cast(tf.cast(relu,tf.bool),tf.float32)
    real = A * flag * fac
    imag = B * flag * fac
    res=tf.concat([real, imag], 3)
    return res


def zRelu(input):
    # input: [batch，行，列，通道]
    relu=tf.nn.relu(input)
    [real,imag]=tf.split(relu,2,3)
    flag_real=tf.cast(tf.cast(real,tf.bool),tf.float32)
    flag_imag=tf.cast(tf.cast(imag,tf.bool),tf.float32)
    flag=flag_imag*flag_real
    real=real*flag
    imag=imag*flag
    return tf.concat([real,imag],3)


def cRelu(input):
    # 相当于独立对实部虚部求relu
    # input: [batch，行，列，通道]
    return tf.nn.relu(input)


def complex_avg_pool(input,ksize,strides):
    # 均值池化
    # input: [batch,行，列，通道]
    # ksize: 池化核描述 [1,行，列，1]
    # strides: 滑动步长 [1,行，列，1]
    return tf.nn.avg_pool(input,ksize,strides,'SAME')


def complex_max_pool(input,ksize,strides):
    # 最大值池化
    # input: [batch,行，列，通道]
    # ksize: 池化核描述 [1,行，列，1]
    # strides: 滑动步长 [1,行，列，1]
    # W=A+Bi
    A, B = tf.split(input, 2, 3)  # 在第3维（通道）上切割
    flatten_A=tf.reshape(A,shape=[-1])
    flatten_B=tf.reshape(B,[-1])

    C = tf.abs(tf.abs(tf.complex(A, B)))
    _, mask = tf.nn.max_pool_with_argmax(C, ksize, strides, padding='SAME')
    output_shape=mask.get_shape()

    flatten_mask=tf.reshape(mask,shape=[-1])
    flatten_real=tf.gather(flatten_A,flatten_mask)
    flatten_imag=tf.gather(flatten_B,flatten_mask)

    real=tf.reshape(flatten_real,output_shape)
    imag=tf.reshape(flatten_imag,output_shape)

    return tf.concat([real,imag],3)


def fully_connect(fc,cur_size,regularizer):
    # 输入：实部+虚部、大小、正则化项
    # 都是2维

    pre_size=tf.cast(fc.get_shape().as_list()[1]/2,tf.int32)
    wr = get_weight([pre_size, cur_size], regularizer)
    wi = get_weight([pre_size, cur_size], regularizer)
    br = get_bias([cur_size])
    bi = get_bias([cur_size])

    [real,imag]=tf.split(fc,2,1) # 第1维度上切成两份
    R=tf.matmul(real,wr)-tf.matmul(imag,wi)+br
    W=tf.matmul(real,wi)+tf.matmul(imag,wr)+bi
    return tf.concat([R,W],1)


def fully_zRelu(fc):
    # 全连接层使用的zRelu
    fc=tf.nn.relu(fc)
    real,imag=tf.split(fc,2,1)
    real_flag=tf.cast(tf.cast(real,tf.bool),tf.float32)
    imag_flag=tf.cast(tf.cast(imag,tf.bool),tf.float32)
    flag=real_flag*imag_flag
    return tf.concat([real*flag,imag*flag],1)

def fully_modRelu(fc):
    # 适用于全连接层的modRelu
    # fc:[batch,units]
    # units前一半是实部，后一半是虚部
    # 返回相同形状

    A, B = tf.split(fc, 2, 1)  # 在第1维切割  A实部 B虚部
    C=tf.complex(A, B)
    C = tf.abs(C)

    # 可以学习的变量b
    b = 50 * np.ones(([C.get_shape()][-1]))
    b = tf.Variable(b, dtype=tf.float32)

    C1 = C - b
    # 衰减系数
    fac = C1 / (C+0.0001)

    flag = tf.nn.relu(C1)
    flag = tf.cast(tf.cast(flag, tf.bool), tf.float32)

    real = A * flag * fac
    imag = B * flag * fac
    return tf.concat([real, imag], 1)


def fully_cRelu(fc):
    # 相当于直接做relu，不区分实部虚部
    return tf.nn.relu(fc)


def dropout(x,keep_prob):
    # 输入x: [batch , size]
    # 以keep_prob的概率留下
    shape=x.get_shape().as_list()
    flag=np.ones(shape=(shape[0],shape[1]))
    for i in range(0,shape[0]):
        for j in range(0,int(shape[1]/2)):
            if np.random.rand() > keep_prob:
                flag[i][j]=0
                flag[i][j+int(shape[1]/2)]=0
    rate=np.mean(flag,1)
    res=np.reshape(rate,(shape[0],1))
    rate=res
    for i in range(shape[1]-1):
        res=np.concatenate([res,rate],1)
    w = tf.convert_to_tensor(flag,dtype=tf.float32)
    res=tf.convert_to_tensor(res,dtype=tf.float32)
    return  w*x/res


def forward(x,train,keep_prob,regularizer):

    # x:前一半通道是实数，后一半通道是复数
    filter1=get_weight([FILTER1_SIZE,FILTER1_SIZE,IMAGE_CHANNEL,FILTER1_NUM],regularizer)
    filter1_bias=get_bias([FILTER1_NUM*2])
    conv1=complex_conv(x,filter1,[1,1,1,1])
    relu1=modRelu(tf.nn.bias_add(conv1,filter1_bias))

    pool1=complex_max_pool(relu1,[1,2,2,1],[1,2,2,1])

    filter2=get_weight([FILTER2_SIZE,FILTER2_SIZE,FILTER1_NUM*2,FILTER2_NUM],regularizer)
    filter2_bias=get_bias([FILTER2_NUM*2])
    conv2=complex_conv(pool1,filter2,[1,1,1,1])
    relu2=modRelu(tf.nn.bias_add(conv2,filter2_bias))

    pool2=complex_max_pool(relu2,[1,2,2,1],[1,2,2,1])

    filter3 = get_weight([FILTER3_SIZE, FILTER3_SIZE, FILTER2_NUM * 2, FILTER3_NUM], regularizer)
    filter3_bias = get_bias([FILTER3_NUM * 2])
    conv3 = complex_conv(pool2, filter3, [1, 1, 1, 1])
    relu3 = modRelu(tf.nn.bias_add(conv3, filter3_bias))

    pool3 = complex_max_pool(relu3, [1, 2, 2, 1], [1, 2, 2, 1])

    pool3_real,pool3_imag=tf.split(pool3,2,3)  #切成实部和虚部
    fc0_ri_shape=pool3_real.get_shape().as_list()
    fc0_size=fc0_ri_shape[1]*fc0_ri_shape[2]*fc0_ri_shape[3]
    fc0_real=tf.reshape(pool3_real,(-1,fc0_size))
    fc0_imag=tf.reshape(pool3_imag,(-1,fc0_size))
    fc0=tf.concat([fc0_real,fc0_imag],1)


    fc1=fully_connect(fc0,FC1_SIZE,regularizer)
    fc1=fully_modRelu(fc1)
    if train:
        fc1 = dropout(fc1,keep_prob)


    fc2 = fully_connect(fc1, FC2_SIZE, regularizer)
    fc2=fully_modRelu(fc2)
    if train:
        fc2=dropout(fc2,keep_prob)


    output = fully_connect(fc2, OUTPUT_NODE, regularizer)
    output_real,output_imag=tf.split(output,2,1)
    return tf.abs(tf.complex(output_real,output_imag))
    #return output_real


# x=tf.Variable(np.random.random((32,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL)),dtype=tf.float32)
# forward(x,True,0.5,0.001)
