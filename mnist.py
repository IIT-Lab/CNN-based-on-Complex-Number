import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True,reshape=True)
# 注意 返回的是展开的 28*28=784 训练时不要忘记reshape一下

train=mnist.train
test=mnist.test

def mnist_fft(xf,batch):
    # xf: batch,784
    # 返回值： y [batch,28,28,2]
    xf=np.reshape(xf,(batch,28,28))
    x=np.ones_like(xf,dtype=np.complex)
    for i in range(0,batch):
        x[i]=np.fft.fftshift(np.fft.fft2(xf[i]))
    # x=np.reshape(x,(batch,28,28,1))
    # y=np.concatenate([np.real(x),np.imag(x)],3)
    y=np.stack([np.real(x),np.imag(x)],3)
    # 不能用绝对值！会改变相位信息
    # y=np.log(np.abs(y)+1)
    return y


