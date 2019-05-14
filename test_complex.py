import tensorflow as tf
import network
import mnist
import numpy as np

BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.005
KEEP_PROB = 1
REGULARIZER = None


def backward(data):
    x = tf.placeholder(tf.float32, (BATCH_SIZE, network.IMAGE_SIZE, network.IMAGE_SIZE, network.IMAGE_CHANNEL))
    y = network.forward(x, False, KEEP_PROB, REGULARIZER)
    y_ = tf.placeholder(tf.float32, (None, network.OUTPUT_NODE))

    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss2 = tf.reduce_mean(loss1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

    saver=tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model = tf.train.latest_checkpoint('mnist_Complex/')
        saver.restore(sess, model)

        accu_sum=0
        loss_sum=0

        for i in range(0,EPOCH):
            xf, yf = data.next_batch(BATCH_SIZE)
            xf = mnist.mnist_fft(xf, BATCH_SIZE)
            accu, los = sess.run([accuracy, loss2], feed_dict={x: xf, y_: yf})
            accu_sum+=accu
            loss_sum+=los

        accu_mean=accu_sum/EPOCH
        loss_mean=loss_sum/EPOCH
        print('loss on test: ',loss_mean )
        print('accuracy on test: ',accu_mean)


backward(mnist.test)