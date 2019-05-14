import tensorflow as tf
import network
import mnist
import matplotlib.pyplot as plt
import random
import numpy as np


BATCH_SIZE=32
EPOCH=500
LEARNING_RATE=0.005
KEEP_PROB=1
REGULARIZER=0.005


def backward(data):
    x=tf.placeholder(tf.float32,(BATCH_SIZE,network.IMAGE_SIZE,network.IMAGE_SIZE,network.IMAGE_CHANNEL))
    y=network.forward(x,True,KEEP_PROB,REGULARIZER)
    y_=tf.placeholder(tf.float32,(None,network.OUTPUT_NODE))

    loss1=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    loss2=tf.reduce_mean(loss1)
    loss3=loss2+tf.add_n(tf.get_collection('losses'))

    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

    opimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss3)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        accu1 = 0
        accu2 = 0

        # 记录要绘图的变量
        x_p = [i for i in range(1, EPOCH + 1)]
        y_loss = [i for i in range(1, EPOCH + 1)]
        y_accu = [i for i in range(1, EPOCH + 1)]

        for i in range(1,EPOCH+1):
            xf,yf=data.next_batch(BATCH_SIZE)

            xf=mnist.mnist_fft(xf,BATCH_SIZE)

            _, accu, los = sess.run([opimizer, accuracy, loss3], feed_dict={x: xf, y_: yf})

            y_loss[i - 1] = los
            y_accu[i - 1] = accu

            if accu1 > 0.75 and accu2 > 0.75 and accu > 0.75:
                saver.save(sess, 'mnist_Complex/mnist_Complex.ckpt',write_meta_graph=False)

            accu1 = accu2
            accu2 = accu

            print('Epoch: ', i)
            print('loss on batch: ', los)
            print('accuracy on batch: ', accu)
            print('.......................................')

    print(y_loss)
    print(y_accu)
    plt.figure()
    plt.plot(x_p[0:len(x_p):4], y_loss[0:len(y_loss):4])
    plt.figure()
    plt.plot(x_p[0:len(x_p):4], y_accu[0:len(y_loss):4])
    plt.show()

backward(mnist.train)





