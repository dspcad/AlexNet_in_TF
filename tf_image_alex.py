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


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      initializer=tf.contrib.layers.xavier_initializer()
      #tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
  )
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def cropImg(target_img, mean_img):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################
  #mean_pixel = [123.182570556, 116.282672124, 103.462011796]
  floating_img = np.empty(target_img.shape, dtype=np.float32)

  #Grayscale Img and convert it to RGB
  if len(target_img.shape) == 2:
    target_img = color.gray2rgb(target_img)


  #floating_img[:,:,0] = target_img[:,:,0] - mean_pixel[0]
  #floating_img[:,:,1] = target_img[:,:,1] - mean_pixel[1]
  #floating_img[:,:,2] = target_img[:,:,2] - mean_pixel[2]

  floating_img = target_img - mean_img
  #floating_img = target_img


  #target_img = target_img - mean_img

  #reflection   = np.random.randint(0,2)
  #if reflection == 0:
  #  target_img = np.fliplr(target_img)


  ################################
  ##      Data Augementation     #
  ################################
  #height_shift = np.random.randint(0,256-224)
  #width_shift  = np.random.randint(0,256-224)

  height_shift = 14
  width_shift  = 14
  target_img = floating_img[height_shift:height_shift+227, width_shift:width_shift+227,:]

  #print target_img
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

    image_class_name = image_name[i].split("_")[0]
    if len(img_batch) == 0:
      img_batch = croppedImg
    else:
      img_batch = np.vstack((img_batch, croppedImg))


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
 
   

  
  img_batch = img_batch.reshape(mini_batch,227,227,3)

  #for i in range(0, mini_batch):
  #  io.imsave("%s_%d.%s" % ("test_img", i, 'jpeg'), img_batch[i])
  #img_batch = img_batch - mean_img
  return img_batch, train_y


def setAsynBatchRead(image_name, class_dict, pool, mean_img):
  batch_idx = np.random.randint(0,len(image_name),mini_batch)
  #batch_idx = np.arange(mini_batch)
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
 
  
  asyn_img_batch = asyn_img_batch.reshape(mini_batch,227,227,3)
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
  #print mean_img

  np.random.seed(31)


  class_dict, image_name  = loadClassName('synset.csv')
  num_images = len(image_name)

  pool = ThreadPool(processes=8)
  print "Multi-threads begin!"


  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 256

  K = 1000 # number of classes
  NUM_FILTER_1 = 48
  NUM_FILTER_2 = 128
  NUM_FILTER_3 = 192
  NUM_FILTER_4 = 192
  NUM_FILTER_5 = 128

  NUM_NEURON_1 = 4096
  NUM_NEURON_2 = 4096

  DROPOUT_PROB_1 = 1.00
  DROPOUT_PROB_2 = 1.00

  LEARNING_RATE = 1e-2
 
  reg = 0 # regularization strength


  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)

  # initialize parameters randomly

  X  = tf.placeholder(tf.float32, shape=[None, 227,227,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1_1  = _variable_with_weight_decay('W1_1', shape=[11, 11, 3, NUM_FILTER_1], stddev=1e-2, wd=5e-4)
  W1_2  = _variable_with_weight_decay('W1_2', shape=[11, 11, 3, NUM_FILTER_1], stddev=1e-2, wd=5e-4)

  W2_1  = _variable_with_weight_decay('W2_1', shape=[5, 5, NUM_FILTER_1,NUM_FILTER_2], stddev=1e-2, wd=5e-4)
  W2_2  = _variable_with_weight_decay('W2_2', shape=[5, 5, NUM_FILTER_1,NUM_FILTER_2], stddev=1e-2, wd=5e-4)

  W3_1  = _variable_with_weight_decay('W3_1', shape=[3, 3, NUM_FILTER_2*2,NUM_FILTER_3], stddev=1e-2, wd=5e-4)
  W3_2  = _variable_with_weight_decay('W3_2', shape=[3, 3, NUM_FILTER_2*2,NUM_FILTER_3], stddev=1e-2, wd=5e-4)

  W4_1  = _variable_with_weight_decay('W4_1', shape=[3, 3, NUM_FILTER_3,NUM_FILTER_4], stddev=1e-2, wd=5e-4)
  W4_2  = _variable_with_weight_decay('W4_2', shape=[3, 3, NUM_FILTER_3,NUM_FILTER_4], stddev=1e-2, wd=5e-4)

  W5_1  = _variable_with_weight_decay('W5_1', shape=[3, 3, NUM_FILTER_4,NUM_FILTER_5], stddev=1e-2, wd=5e-4)
  W5_2  = _variable_with_weight_decay('W5_2', shape=[3, 3, NUM_FILTER_4,NUM_FILTER_5], stddev=1e-2, wd=5e-4)

  W6    = _variable_with_weight_decay('W6', shape=[6*6*NUM_FILTER_5*2,NUM_NEURON_1], stddev=5e-3, wd=5e-4)

  W7    = _variable_with_weight_decay('W7', shape=[NUM_NEURON_1,NUM_NEURON_2], stddev=5e-3, wd=5e-4)

  W8    = _variable_with_weight_decay('W8', shape=[NUM_NEURON_2,K], stddev=1e-2, wd=5e-4)
  

  #W1_1  = tf.Variable(tf.truncated_normal([11,11,3,NUM_FILTER_1], stddev=0.01))
  #W1_2  = tf.Variable(tf.truncated_normal([11,11,3,NUM_FILTER_1], stddev=0.01))

  #W2_1  = tf.Variable(tf.truncated_normal([5,5,NUM_FILTER_1,NUM_FILTER_2], stddev=0.01))
  #W2_2  = tf.Variable(tf.truncated_normal([5,5,NUM_FILTER_1,NUM_FILTER_2], stddev=0.01))

  #W3_1  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2*2,NUM_FILTER_3], stddev=0.01))
  #W3_2  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2*2,NUM_FILTER_3], stddev=0.01))

  #W4_1  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.01))
  #W4_2  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.01))

  #W5_1  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.01))
  #W5_2  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.01))

  #W6    = tf.Variable(tf.truncated_normal([6*6*NUM_FILTER_5*2,NUM_NEURON_1], stddev=0.005))

  #W7    = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.005))

  #W8    = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.01))


  #W1_1 = tf.get_variable("W1_1", shape=[11,11,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())
  #W1_2 = tf.get_variable("W1_2", shape=[11,11,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())

  #W2_1 = tf.get_variable("W2_1", shape=[5,5,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())
  #W2_2 = tf.get_variable("W2_2", shape=[5,5,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())

  #W3_1 = tf.get_variable("W3_1", shape=[3,3,NUM_FILTER_2*2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())
  #W3_2 = tf.get_variable("W3_2", shape=[3,3,NUM_FILTER_2*2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())

  #W4_1 = tf.get_variable("W4_1", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())
  #W4_2 = tf.get_variable("W4_2", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())

  #W5_1 = tf.get_variable("W5_1", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())
  #W5_2 = tf.get_variable("W5_2", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())

  #W6   = tf.get_variable("W6", shape=[6*6*NUM_FILTER_5*2,NUM_NEURON_1], initializer=tf.contrib.layers.xavier_initializer())

  #W7   = tf.get_variable("W7", shape=[NUM_NEURON_1,NUM_NEURON_2], initializer=tf.contrib.layers.xavier_initializer())

  #W8   = tf.get_variable("W8", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer())


  b1_1 = tf.Variable(tf.zeros([NUM_FILTER_1]))
  b1_2 = tf.Variable(tf.zeros([NUM_FILTER_1]))

  b2_1 = tf.Variable(tf.ones([NUM_FILTER_2])/10)
  b2_2 = tf.Variable(tf.ones([NUM_FILTER_2])/10)

  b3_1 = tf.Variable(tf.zeros([NUM_FILTER_3]))
  b3_2 = tf.Variable(tf.zeros([NUM_FILTER_3]))

  b4_1 = tf.Variable(tf.ones([NUM_FILTER_4])/10)
  b4_2 = tf.Variable(tf.ones([NUM_FILTER_4])/10)

  b5_1 = tf.Variable(tf.ones([NUM_FILTER_5])/10)
  b5_2 = tf.Variable(tf.ones([NUM_FILTER_5])/10)

  b6   = tf.Variable(tf.ones([NUM_NEURON_1])/10)

  b7   = tf.Variable(tf.ones([NUM_NEURON_2])/10)

  b8 = tf.Variable(tf.zeros([K]))




  ##########################
  #      Architecture      #
  ##########################
  #===== Layer 1 =====#
  conv1_1 = tf.nn.relu(tf.nn.conv2d(X,  W1_1, strides=[1,4,4,1], padding='SAME')+b1_1)
  norm1_1 = tf.nn.lrn(conv1_1, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool1_1 = tf.nn.max_pool(norm1_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  conv1_2 = tf.nn.relu(tf.nn.conv2d(X,  W1_2, strides=[1,4,4,1], padding='SAME')+b1_2)
  norm1_2 = tf.nn.lrn(conv1_2, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool1_2 = tf.nn.max_pool(norm1_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')


  #===== Layer 2 =====#
  conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1_1, W2_1, strides=[1,1,1,1], padding='SAME')+b2_1)
  norm2_1 = tf.nn.lrn(conv2_1, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool2_1 = tf.nn.max_pool(norm2_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  conv2_2 = tf.nn.relu(tf.nn.conv2d(pool1_2, W2_2, strides=[1,1,1,1], padding='SAME')+b2_2)
  norm2_2 = tf.nn.lrn(conv2_2, alpha=1e-4, beta=0.75, depth_radius=5, bias=2.0)
  pool2_2 = tf.nn.max_pool(norm2_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  
  cross2 = tf.concat([pool2_1, pool2_2], 3)


  #===== Layer 3 =====#
  conv3_1 = tf.nn.relu(tf.nn.conv2d(cross2, W3_1, strides=[1,1,1,1], padding='SAME')+b3_1)
  conv3_2 = tf.nn.relu(tf.nn.conv2d(cross2, W3_2, strides=[1,1,1,1], padding='SAME')+b3_2)


  #===== Layer 4 =====#
  conv4_1 = tf.nn.relu(tf.nn.conv2d(conv3_1, W4_1, strides=[1,1,1,1], padding='SAME')+b4_1)
  conv4_2 = tf.nn.relu(tf.nn.conv2d(conv3_2, W4_2, strides=[1,1,1,1], padding='SAME')+b4_2)

  #===== Layer 5 =====#
  conv5_1 = tf.nn.relu(tf.nn.conv2d(conv4_1, W5_1, strides=[1,1,1,1], padding='SAME')+b5_1)
  conv5_2 = tf.nn.relu(tf.nn.conv2d(conv4_2, W5_2, strides=[1,1,1,1], padding='SAME')+b5_2)

  pool5_1 = tf.nn.max_pool(conv5_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
  pool5_2 = tf.nn.max_pool(conv5_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')


  cross5 = tf.concat([pool5_1, pool5_2], 1)
  YY = tf.reshape(cross5, shape=[-1,6*6*NUM_FILTER_5*2])

  #===== Layer 6 =====#
  fc1 = tf.nn.relu(tf.matmul(YY,W6)+b6)
  #fc1_drop = tf.nn.dropout(fc1, keep_prob_1)

  #===== Layer 7 =====#
  fc2 = tf.nn.relu(tf.matmul(fc1,W7)+b7)
  #fc2_drop = tf.nn.dropout(fc2, keep_prob_1)

  #===== Layer 8 =====#
  Y  = tf.nn.softmax(tf.matmul(fc2,W8)+b8)



  for var in tf.trainable_variables():
    print var

  #diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
  tf.add_to_collection('losses', cross_entropy)
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
 

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                             1000000, 0.9, staircase=True)

  train_step = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(total_loss, global_step=global_step)
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
  #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

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
  epoch_num     = 0
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
      if itr % 10 == 0:
        print "Iter %d:  learning rate: %f  dropout: (%.1f %.1f) cross entropy: %f total loss: %f  accuracy: %f" % (itr,
                                                                learning_rate.eval(feed_dict={X: x, Y_: y, 
                                                                                              keep_prob_1: DROPOUT_PROB_1, 
                                                                                              keep_prob_2: DROPOUT_PROB_2}),
                                                                DROPOUT_PROB_1,
                                                                DROPOUT_PROB_2,
                                                                cross_entropy.eval(feed_dict={X: x, Y_: y, 
                                                                                                            keep_prob_1: DROPOUT_PROB_1, 
                                                                                                            keep_prob_2: DROPOUT_PROB_2}),
                                                                total_loss.eval(feed_dict={X: x, Y_: y, 
                                                                                                            keep_prob_1: DROPOUT_PROB_1, 
                                                                                                            keep_prob_2: DROPOUT_PROB_2}),

                                                                accuracy.eval(feed_dict={X: x, Y_: y, 
                                                                                                       keep_prob_1: DROPOUT_PROB_1, 
                                                                                                       keep_prob_2: DROPOUT_PROB_2}))



      x = getAsynBatchRead(asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7)
      y = asyn_train_y

      #elapsed_time = time.time() - start_time
      #print "Time for async read and training: %f" % elapsed_time
      

      #print train_step
 
      #print "W9:"
      #print sess.run(W9) 
      if itr % 1000 == 0 and itr != 0:
        model_name = "./checkpoint/model_%d.ckpt" % itr
        save_path = saver.save(sess, model_name)
        #save_path = saver.save(sess, "./checkpoint/model.ckpt")
        print("Model saved in file: %s" % save_path)


      if epoch_counter*mini_batch > num_images:
        epoch_counter = 0
        epoch_num = epoch_num + 1
        print "Epoch: ", epoch_num
      else:
        epoch_counter = epoch_counter + 1






