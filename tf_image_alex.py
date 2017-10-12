#!/usr/bin/python

import cPickle
import numpy as np
import os
import csv
from skimage import io
from skimage import transform
from skimage import color
import tensorflow as tf
from multiprocessing.pool import ThreadPool
import time
#import matplotlib.pyplot as plt

def cropImg(target_img, mean_img):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  #Grayscale Img and convert it to RGB
  if len(target_img.shape) == 2:
    target_img = color.gray2rgb(target_img)


  #if target_img.shape[0] < target_img.shape[1]:
  #  width = int(target_img.shape[1]*256/target_img.shape[0])
  #  #if width < 224:
  #  #  width = 224

  #  offset = int((width-256)/2)

  #  target_img = transform.resize(target_img, (256,width,3))
  #  target_img = target_img[:, offset:256+offset, :]
  #else:
  #  height = int(target_img.shape[0]*256/target_img.shape[1])
  #  #if height < 224:
  #  #  height = 224

  #  offset = int((height-256)/2)

  #  target_img = transform.resize(target_img, (height,256,3))
  #  target_img = target_img[offset:256+offset, :, :]


  target_img = target_img - mean_img

  reflection   = np.random.randint(0,2)
  if reflection == 0:
    target_img = np.fliplr(target_img)


  ###############################
  #      Data Augementation     #
  ###############################
  height_shift = np.random.randint(0,256-224)
  width_shift  = np.random.randint(0,256-224)

  target_img = target_img[height_shift:height_shift+224, width_shift:width_shift+224,:]


  return target_img


def batchCroppedImgRead(thread_name, dirpath, image_name, mean_img, partial_batch_idx):
  #print "%s is cropping the images..." % thread_name
  img_batch = []

  for i in partial_batch_idx:
    absfile = os.path.join(dirpath, image_name[i])
    target_img = io.imread(absfile)

    #################################
    # convert RGB from float to int #
    #################################
    croppedImg = cropImg(target_img, mean_img)
    #croppedImg = cropImg(target_img, mean_img)*255
    #croppedImg = croppedImg.astype('uint8') 
    #croppedImg[:,:,0] = croppedImg[:,:,0] - VGG_MEAN[0]
    #croppedImg[:,:,1] = croppedImg[:,:,1] - VGG_MEAN[1]
    #croppedImg[:,:,2] = croppedImg[:,:,2] - VGG_MEAN[2]
    #io.imsave("%s_%d_%d.%s" % ("crop_img", i, int(class_dict[image_class_name]), 'jpeg'), target_img)

    image_class_name = image_name[i].split("_")[0]
    if len(img_batch) == 0:
      img_batch = croppedImg
    else:
      img_batch = np.vstack((img_batch, croppedImg))

  #test_img_batch = img_batch.reshape(len(partial_batch_idx),224,224,3)
  #for i in range(0,len(partial_batch_idx)):
  #  io.imsave("%s_%d.%s" % ("crop_img", partial_batch_idx[i], 'jpeg'), test_img_batch[i])

  return img_batch

def batchRead(image_name, class_dict, mean_img, pool):

  batch_idx = np.random.randint(0,len(image_name),mini_batch)
  #batch_idx = np.arange(mini_batch)
  #dirpath = '/home/hhwu/ImageNet/train/'
  dirpath = '/mnt/ramdisk/crop_train/'


  #convert to one hot labels
  train_y = np.zeros((mini_batch,K))
  #print class_dict
  for i in range(0, len(batch_idx)):
    image_class_name = image_name[batch_idx[i]].split("_")[0]
    #print i
    #print image_class_name
    #print class_dict[image_class_name]
    train_y[i][int(class_dict[image_class_name])] = 1

    #print "test_y[%d][%d] = %d" % (i,int(class_dict[image_class_name]),test_y[i][int(class_dict[image_class_name])])

  #img_batch = batchCroppedImgRead("Thread-0", dirpath, image_name, batch_idx)

  async_result_0 = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, image_name, mean_img, batch_idx[:int(mini_batch/8)]))
  async_result_1 = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, image_name, mean_img, batch_idx[int(mini_batch/8):int(2*mini_batch/8)]))
  async_result_2 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, image_name, mean_img, batch_idx[int(2*mini_batch/8):int(3*mini_batch/8)]))
  async_result_3 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, image_name, mean_img, batch_idx[int(3*mini_batch/8):int(4*mini_batch/8)]))
  async_result_4 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, image_name, mean_img, batch_idx[int(4*mini_batch/8):int(5*mini_batch/8)]))
  async_result_5 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, image_name, mean_img, batch_idx[int(5*mini_batch/8):int(6*mini_batch/8)]))
  async_result_6 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, image_name, mean_img, batch_idx[int(6*mini_batch/8):int(7*mini_batch/8)]))
  async_result_7 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, image_name, mean_img, batch_idx[int(7*mini_batch/8):]))

  img_batch    = async_result_0.get()
  return_val_1 = async_result_1.get()
  return_val_2 = async_result_2.get()
  return_val_3 = async_result_3.get()
  return_val_4 = async_result_4.get()
  return_val_5 = async_result_5.get()
  return_val_6 = async_result_6.get()
  return_val_7 = async_result_7.get()


  img_batch = np.vstack((img_batch, return_val_1))
  img_batch = np.vstack((img_batch, return_val_2))
  img_batch = np.vstack((img_batch, return_val_3))
  img_batch = np.vstack((img_batch, return_val_4))
  img_batch = np.vstack((img_batch, return_val_5))
  img_batch = np.vstack((img_batch, return_val_6))
  img_batch = np.vstack((img_batch, return_val_7))
 
   

  #async_result_0  = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, image_name, batch_idx[:int(mini_batch/16)]))
  #async_result_1  = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, image_name, batch_idx[int(mini_batch/16):int(2*mini_batch/16)]))
  #async_result_2  = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, image_name, batch_idx[int(2*mini_batch/16):int(3*mini_batch/16)]))
  #async_result_3  = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, image_name, batch_idx[int(3*mini_batch/16):int(4*mini_batch/16)]))
  #async_result_4  = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, image_name, batch_idx[int(4*mini_batch/16):int(5*mini_batch/16)]))
  #async_result_5  = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, image_name, batch_idx[int(5*mini_batch/16):int(6*mini_batch/16)]))
  #async_result_6  = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, image_name, batch_idx[int(6*mini_batch/16):int(7*mini_batch/16)]))
  #async_result_7  = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, image_name, batch_idx[int(7*mini_batch/16):int(8*mini_batch/16)]))
  #async_result_8  = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, image_name, batch_idx[int(8*mini_batch/16):int(9*mini_batch/16)]))
  #async_result_9  = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, image_name, batch_idx[int(9*mini_batch/16):int(10*mini_batch/16)]))
  #async_result_10 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, image_name, batch_idx[int(10*mini_batch/16):int(11*mini_batch/16)]))
  #async_result_11 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, image_name, batch_idx[int(11*mini_batch/16):int(12*mini_batch/16)]))
  #async_result_12 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, image_name, batch_idx[int(12*mini_batch/16):int(13*mini_batch/16)]))
  #async_result_13 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, image_name, batch_idx[int(13*mini_batch/16):int(14*mini_batch/16)]))
  #async_result_14 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, image_name, batch_idx[int(14*mini_batch/16):int(15*mini_batch/16)]))
  #async_result_15 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, image_name, batch_idx[int(15*mini_batch/16):]))

  #img_batch     = async_result_0.get()
  #return_val_1  = async_result_1.get()
  #return_val_2  = async_result_2.get()
  #return_val_3  = async_result_3.get()
  #return_val_4  = async_result_4.get()
  #return_val_5  = async_result_5.get()
  #return_val_6  = async_result_6.get()
  #return_val_7  = async_result_7.get()
  #return_val_8  = async_result_8.get()
  #return_val_9  = async_result_9.get()
  #return_val_10 = async_result_10.get()
  #return_val_11 = async_result_11.get()
  #return_val_12 = async_result_12.get()
  #return_val_13 = async_result_13.get()
  #return_val_14 = async_result_14.get()
  #return_val_15 = async_result_15.get()

  #img_batch = np.vstack((img_batch, return_val_1))
  #img_batch = np.vstack((img_batch, return_val_2))
  #img_batch = np.vstack((img_batch, return_val_3))
  #img_batch = np.vstack((img_batch, return_val_4))
  #img_batch = np.vstack((img_batch, return_val_5))
  #img_batch = np.vstack((img_batch, return_val_6))
  #img_batch = np.vstack((img_batch, return_val_7))
  #img_batch = np.vstack((img_batch, return_val_8))
  #img_batch = np.vstack((img_batch, return_val_9))
  #img_batch = np.vstack((img_batch, return_val_10))
  #img_batch = np.vstack((img_batch, return_val_11))
  #img_batch = np.vstack((img_batch, return_val_12))
  #img_batch = np.vstack((img_batch, return_val_13))
  #img_batch = np.vstack((img_batch, return_val_14))
  #img_batch = np.vstack((img_batch, return_val_15))
 
  
  img_batch = img_batch.reshape(mini_batch,224,224,3)

  #for i in range(0, mini_batch):
  #  io.imsave("%s_%d.%s" % ("test_img", i, 'jpeg'), img_batch[i])
  #print img_batch[0]
  #print class_dict


  #return img_batch, label_batch
  #return img_batch, train_y

  #img_batch = img_batch - mean_img
  return img_batch, train_y


def setAsynBatchRead(image_name, class_dict, pool, mean_img):
  batch_idx = np.random.randint(0,len(image_name),mini_batch)
  dirpath = '/mnt/ramdisk/crop_train/'


  #convert to one hot labels
  train_y = np.zeros((mini_batch,K))
  for i in range(0, len(batch_idx)):
    image_class_name = image_name[batch_idx[i]].split("_")[0]
    train_y[i][int(class_dict[image_class_name])] = 1



  async_result_0 = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, image_name, mean_img, batch_idx[:int(mini_batch/8)]))
  async_result_1 = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, image_name, mean_img, batch_idx[int(mini_batch/8):int(2*mini_batch/8)]))
  async_result_2 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, image_name, mean_img, batch_idx[int(2*mini_batch/8):int(3*mini_batch/8)]))
  async_result_3 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, image_name, mean_img, batch_idx[int(3*mini_batch/8):int(4*mini_batch/8)]))
  async_result_4 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, image_name, mean_img, batch_idx[int(4*mini_batch/8):int(5*mini_batch/8)]))
  async_result_5 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, image_name, mean_img, batch_idx[int(5*mini_batch/8):int(6*mini_batch/8)]))
  async_result_6 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, image_name, mean_img, batch_idx[int(6*mini_batch/8):int(7*mini_batch/8)]))
  async_result_7 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, image_name, mean_img, batch_idx[int(7*mini_batch/8):]))

  return async_result_0, async_result_1, async_result_2, async_result_3, async_result_4, async_result_5, async_result_6, async_result_7, train_y


def getAsynBatchRead(async_result_0, async_result_1, async_result_2, async_result_3, async_result_4, async_result_5, async_result_6, async_result_7):
  asyn_img_batch = async_result_0.get()
  return_val_1   = async_result_1.get()
  return_val_2   = async_result_2.get()
  return_val_3   = async_result_3.get()
  return_val_4   = async_result_4.get()
  return_val_5   = async_result_5.get()
  return_val_6   = async_result_6.get()
  return_val_7   = async_result_7.get()


  asyn_img_batch = np.vstack((asyn_img_batch, return_val_1))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_2))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_3))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_4))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_5))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_6))
  asyn_img_batch = np.vstack((asyn_img_batch, return_val_7))
 
  
  asyn_img_batch = asyn_img_batch.reshape(mini_batch,224,224,3)
  #asyn_img_batch = asyn_img_batch - mean_img

  return asyn_img_batch





def loadClassName(filename):
  class_name = []
  with open(filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for row in spamreader:
      class_name.append(row[1])
      i = i+1
      if i >= 1000:
        break

  class_dict = {}
  for i in xrange(0,len(class_name)):
    class_dict[class_name[i].replace('\'', '')] = i


  image_name = []
  for dirpath, dirnames, filenames in os.walk('/mnt/ramdisk/crop_train/'):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    print "The number of files: %d" % len(filenames)

    image_name = filenames

  print "The number of classes: %d" % len(class_name)
  return class_dict, image_name



if __name__ == '__main__':
  print '===== Start loading the mean of ILSVRC2012 ====='

  fo = open('mean.bin', 'rb')
  mean_img = cPickle.load(fo)
  fo.close()
  print mean_img
  #mean_img = mean_img*255
  #mean_img = mean_img.astype('uint8')


  class_dict, image_name  = loadClassName('synset.csv')

  pool = ThreadPool(processes=8)
  print "Multi-threads begin!"


  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128

  K = 1000 # number of classes
  NUM_FILTER_1 = 48
  NUM_FILTER_2 = 128
  NUM_FILTER_3 = 192
  NUM_FILTER_4 = 192
  NUM_FILTER_5 = 128

  NUM_NEURON_1 = 2048
  NUM_NEURON_2 = 2048

  DROPOUT_PROB_1 = 1.00
  DROPOUT_PROB_2 = 1.00

  LEARNING_RATE = 1e-2
 
  reg = 1e-3 # regularization strength


  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)

  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 224,224,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])


  W1  = tf.Variable(tf.truncated_normal([11,11,3,NUM_FILTER_1], stddev=0.01))
  W2  = tf.Variable(tf.truncated_normal([5,5,NUM_FILTER_1,NUM_FILTER_2], stddev=0.01))
  W3  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=0.01))
  W4  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.01))
  W5  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.01))
  W6 = tf.Variable(tf.truncated_normal([6*6*NUM_FILTER_5,NUM_NEURON_1], stddev=0.01))
  W7 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.01))
  W8 = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.01))


  b1  = tf.Variable(tf.ones([NUM_FILTER_1]))
  b2  = tf.Variable(tf.zeros([NUM_FILTER_2]))
  b3  = tf.Variable(tf.ones([NUM_FILTER_3]))
  b4  = tf.Variable(tf.zeros([NUM_FILTER_4]))
  b5  = tf.Variable(tf.zeros([NUM_FILTER_5]))
  b6 = tf.Variable(tf.ones([NUM_NEURON_1]))
  b7 = tf.Variable(tf.ones([NUM_NEURON_2]))
  b8 = tf.Variable(tf.ones([K]))


  #===== architecture =====#
  conv1 = tf.nn.relu(tf.nn.conv2d(X,  W1, strides=[1,4,4,1], padding='SAME')+b1)
  norm1 = tf.nn.lrn(conv1, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool1 = tf.nn.max_pool(norm1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  norm2 = tf.nn.lrn(conv2, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides=[1,1,1,1], padding='SAME')+b3)
  conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
  conv5 = tf.nn.relu(tf.nn.conv2d(conv4, W5, strides=[1,1,1,1], padding='SAME')+b5)
  pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  YY = tf.reshape(pool5, shape=[-1,6*6*NUM_FILTER_5])

  fc1 = tf.nn.relu(tf.matmul(YY,W6)+b6)
  #fc1_drop = tf.nn.dropout(fc1, keep_prob_1)

  fc2 = tf.nn.relu(tf.matmul(fc1,W7)+b7)
  #fc2_drop = tf.nn.dropout(fc2, keep_prob_1)

  Y  = tf.nn.softmax(tf.matmul(fc2,W8)+b8)





  diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)

  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = LEARNING_RATE
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             1000000, 0.9, staircase=True)
  train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


  correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





  # Passing global_step to minimize() will increment it at each step.
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
 

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver() 

  #learning_rate = tf.placeholder(tf.float32, shape=[])
  #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


  # Restore variables from disk.
  #saver.restore(sess, "./checkpoint/model_2990000.ckpt")
  #print("Model restored.")

  #te_x, te_y = batchTestRead(te_data10, te_labels10)
  print '  Start training... '
  idx_start = 0
  epoch_counter = 0

  max_test_acc = 0
  #num_input_data =tr_data10.shape[0]
  #x, y = batchRead(image_name, class_dict, pool)

  #for i in range(0, mini_batch):
  #  io.imsave("%s_%d.%s" % ("test_img", i, 'jpeg'), x[i])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x, y = batchRead(image_name, class_dict, mean_img, pool)
    for itr in xrange(1000000):
      #x, y = batchRead(image_name, class_dict, mean_img, pool)


      #print y
      asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7, asyn_train_y = setAsynBatchRead(image_name, class_dict, pool, mean_img)
      #start_time = time.time()
      train_step.run(feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1, keep_prob_2: DROPOUT_PROB_2})
      #elapsed_time = time.time() - start_time
      #print "Time for async read and training: %f" % elapsed_time
      x = getAsynBatchRead(asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7)
      y = asyn_train_y
      

      #print train_step
 
      #print "W9:"
      #print sess.run(W9) 
      if itr % 10 == 0:
        print "Iter %d:  learning rate: %f  dropout: (%.1f %.1f) cross entropy: %f  accuracy: %f" % (itr,
                                                                learning_rate.eval(feed_dict={X: x, Y_: y, 
                                                                                              keep_prob_1: DROPOUT_PROB_1, 
                                                                                              keep_prob_2: DROPOUT_PROB_2}),
                                                                DROPOUT_PROB_1,
                                                                DROPOUT_PROB_2,
                                                                cross_entropy.eval(feed_dict={X: x, Y_: y, 
                                                                                                            keep_prob_1: DROPOUT_PROB_1, 
                                                                                                            keep_prob_2: DROPOUT_PROB_2}),
                                                                accuracy.eval(feed_dict={X: x, Y_: y, 
                                                                                                       keep_prob_1: DROPOUT_PROB_1, 
                                                                                                       keep_prob_2: DROPOUT_PROB_2}))

      if itr % 1000 == 0 and itr != 0:
        model_name = "./checkpoint/model_%d.ckpt" % itr
        save_path = saver.save(sess, model_name)
        #save_path = saver.save(sess, "./checkpoint/model.ckpt")
        print("Model saved in file: %s" % save_path)


