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
from PIL import Image
import train_util as tu
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
      #initializer=tf.contrib.layers.xavier_initializer()
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
  )
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def cropImg(target_img):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################
  #mean_pixel = [123.182570556, 116.282672124, 103.462011796]
  mean_pixel = [123.68, 116.779, 103.939]
  #floating_img = np.empty(target_img.shape, dtype=np.float32)


  #Grayscale Img and convert it to RGB
  #if len(target_img.shape) == 2:
  #  target_img = color.gray2rgb(target_img)


  if target_img.size[0] < target_img.size[1]:
    h = int(target_img.size[1]*256/target_img.size[0])
    #if width < 224:
    #  width = 224

    #target_img = transform.resize(target_img, (256,width,3))
    target_img = target_img.resize((256,h), Image.ANTIALIAS)
  else:
    w = int(target_img.size[0]*256/target_img.size[1])
    #if height < 224:
    #  height = 224

    target_img = target_img.resize((w,256), Image.ANTIALIAS)

  #print target_img.shape
  x = np.random.randint(0, target_img.size[0] - 224)
  y = np.random.randint(0, target_img.size[1] - 224)

  img_cropped = target_img.crop((x, y, x + 224, y + 224))
  #target_img = target_img[x:x+224,y:y+224,:]

  #target_img = target_img*255
  #target_img = target_img.astype('uint8')
  #print img_cropped


  floating_img = np.array(img_cropped, dtype=np.float32)
  floating_img[:,:,0] -= mean_pixel[0]
  floating_img[:,:,1] -= mean_pixel[1]
  floating_img[:,:,2] -= mean_pixel[2]
 
  #print floating_img
  ###############################
  #      Data Augementation     #
  ###############################
  #reflection   = np.random.randint(0,2)
  #if reflection == 0:
  #  target_img = np.fliplr(target_img)




  return floating_img


def batchCroppedImgRead(thread_name, dirpath, class_name, partial_batch_idx):
  #print "%s is cropping the images..." % thread_name
  img_batch = []

  for i in partial_batch_idx:
    images_folder = os.path.join(dirpath, class_name[i])
    absfile = os.path.join(images_folder, np.random.choice(os.listdir(images_folder)))
    #target_img = io.imread(absfile)
    target_img = Image.open(absfile).convert('RGB')
    #print target_img.shape

    #################################
    # convert RGB from float to int #
    #################################
    croppedImg = cropImg(target_img)
   
    if len(img_batch) == 0:
      img_batch = croppedImg
    else:
      img_batch = np.vstack((img_batch, croppedImg))

  #test_img_batch = img_batch.reshape(len(partial_batch_idx),224,224,3)
  #for i in range(0,len(partial_batch_idx)):
  #  io.imsave("%s_%d.%s" % ("crop_img", partial_batch_idx[i], 'jpeg'), test_img_batch[i])

  return img_batch

def batchRead(class_name, pool):
  batch_idx = np.random.randint(0, K, size=mini_batch)
  #print batch_idx
  #batch_idx = np.arange(mini_batch)
  #dirpath = '/home/hhwu/ImageNet/train/'
  dirpath = '/mnt/ramdisk/crop_train/'

  #convert to one hot labels
  train_y = np.zeros((mini_batch,K))
  #print class_dict
  #for i in range(0, len(batch_idx)):
  for i in range(0, mini_batch):
    train_y[i][batch_idx[i]] = 1
    #print "batch_idx[%d]: %d" % (i,batch_idx[i])
    #print "train[%d][%d]: %d" % (i,batch_idx[i], train_y[i][batch_idx[i]])

    #print "test_y[%d][%d] = %d" % (i,int(class_dict[image_class_name]),test_y[i][int(class_dict[image_class_name])])

  #img_batch = batchCroppedImgRead("Thread-0", dirpath, class_name, batch_idx)

  async_result_0 = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, class_name, batch_idx[:int(mini_batch/8)]))
  async_result_1 = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, class_name, batch_idx[int(mini_batch/8):int(2*mini_batch/8)]))
  async_result_2 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, class_name, batch_idx[int(2*mini_batch/8):int(3*mini_batch/8)]))
  async_result_3 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, class_name, batch_idx[int(3*mini_batch/8):int(4*mini_batch/8)]))
  async_result_4 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, class_name, batch_idx[int(4*mini_batch/8):int(5*mini_batch/8)]))
  async_result_5 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, class_name, batch_idx[int(5*mini_batch/8):int(6*mini_batch/8)]))
  async_result_6 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, class_name, batch_idx[int(6*mini_batch/8):int(7*mini_batch/8)]))
  async_result_7 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, class_name, batch_idx[int(7*mini_batch/8):]))

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


def setAsynBatchRead(class_name, pool):
  batch_idx = np.random.randint(0, K, size=mini_batch)
  #batch_idx = np.random.randint(0, len(image_name), size=mini_batch)
  #print batch_idx

  #batch_idx = np.arange(mini_batch)
  dirpath = '/mnt/ramdisk/crop_train/'
  #dirpath = '/home/hhwu/ImageNet/train/'


  #convert to one hot labels
  train_y = np.zeros((mini_batch,K))
  for i in range(0, len(batch_idx)):
    train_y[i][batch_idx[i]] = 1


  async_result_0 = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, class_name, batch_idx[:int(mini_batch/8)]))
  async_result_1 = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, class_name, batch_idx[int(mini_batch/8):int(2*mini_batch/8)]))
  async_result_2 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, class_name, batch_idx[int(2*mini_batch/8):int(3*mini_batch/8)]))
  async_result_3 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, class_name, batch_idx[int(3*mini_batch/8):int(4*mini_batch/8)]))
  async_result_4 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, class_name, batch_idx[int(4*mini_batch/8):int(5*mini_batch/8)]))
  async_result_5 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, class_name, batch_idx[int(5*mini_batch/8):int(6*mini_batch/8)]))
  async_result_6 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, class_name, batch_idx[int(6*mini_batch/8):int(7*mini_batch/8)]))
  async_result_7 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, class_name, batch_idx[int(7*mini_batch/8):]))

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
      class_name.append(row[1].replace('\'', ''))
      i = i+1
      if i >= 1000:
        break




  #print class_name
  print "The number of classes: %d" % len(class_name)

  return class_name



if __name__ == '__main__':
  print '===== Start loading the mean of ILSVRC2012 ====='

  fo = open('mean.bin', 'rb')
  mean_img = cPickle.load(fo)
  fo.close()
  #print mean_img
  #mean_img = mean_img*255
  #mean_img = mean_img.astype('uint8')

  np.random.seed(31)

  class_name  = loadClassName('synset.csv')

  pool = ThreadPool(processes=8)
  print "Multi-threads begin!"


  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 256

  K = 1000 # number of classes
  NUM_FILTER_1 = 96
  NUM_FILTER_2 = 256
  NUM_FILTER_3 = 384
  NUM_FILTER_4 = 384
  NUM_FILTER_5 = 256

  NUM_NEURON_1 = 4096
  NUM_NEURON_2 = 4096

  DROPOUT_PROB_1 = 0.50
  DROPOUT_PROB_2 = 0.50

  LEARNING_RATE = 1e-2
 
  LMBDA    = 5e-4

  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)


  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 224,224,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])


  #W1 = tf.get_variable("W1", shape=[11,11,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())
  #W2 = tf.get_variable("W2", shape=[5,5,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())
  #W3 = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())
  #W4 = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())
  #W5 = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())
  #W6 = tf.get_variable("W6", shape=[6*6*NUM_FILTER_5,NUM_NEURON_1], initializer=tf.contrib.layers.xavier_initializer())
  #W7 = tf.get_variable("W7", shape=[NUM_NEURON_1,NUM_NEURON_2], initializer=tf.contrib.layers.xavier_initializer())
  #W8 = tf.get_variable("W8", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer())


  #W1  = _variable_with_weight_decay('W1', shape=[11, 11, 3, NUM_FILTER_1], stddev=1e-2, wd=5e-4)
  #W2  = _variable_with_weight_decay('W2', shape=[5, 5, NUM_FILTER_1,NUM_FILTER_2], stddev=1e-2, wd=5e-4)
  #W3  = _variable_with_weight_decay('W3', shape=[3, 3, NUM_FILTER_2,NUM_FILTER_3], stddev=1e-2, wd=5e-4)
  #W4  = _variable_with_weight_decay('W4', shape=[3, 3, NUM_FILTER_3,NUM_FILTER_4], stddev=1e-2, wd=5e-4)
  #W5  = _variable_with_weight_decay('W5', shape=[3, 3, NUM_FILTER_4,NUM_FILTER_5], stddev=1e-2, wd=5e-4)
  #W6  = _variable_with_weight_decay('W6', shape=[6*6*NUM_FILTER_5,NUM_NEURON_1], stddev=1e-2, wd=5e-4)
  #W7  = _variable_with_weight_decay('W7', shape=[NUM_NEURON_1,NUM_NEURON_2], stddev=1e-2, wd=5e-4)
  #W8  = _variable_with_weight_decay('W8', shape=[NUM_NEURON_2,K], stddev=1e-2, wd=5e-4)

  W1 = tf.Variable(tf.truncated_normal([11,11,3,NUM_FILTER_1], stddev=0.01), name='W1')
  W2 = tf.Variable(tf.truncated_normal([5,5,NUM_FILTER_1,NUM_FILTER_2], stddev=0.01), name='W2')
  W3 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=0.01), name='W3')
  W4 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.01), name='W4')
  W5 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.01), name='W5')
  W6 = tf.Variable(tf.truncated_normal([6*6*NUM_FILTER_5,NUM_NEURON_1], stddev=0.01), name='W6')
  W7 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.01), name='W7')
  W8 = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.01), name='W8')

  tf.add_to_collection('weights', W1)
  tf.add_to_collection('weights', W2)
  tf.add_to_collection('weights', W3)
  tf.add_to_collection('weights', W4)
  tf.add_to_collection('weights', W5)
  tf.add_to_collection('weights', W6)
  tf.add_to_collection('weights', W7)
  tf.add_to_collection('weights', W8)

  b1 = tf.Variable(tf.constant(0.0, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
  b2 = tf.Variable(tf.constant(1.0, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
  b3 = tf.Variable(tf.constant(0.0, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
  b4 = tf.Variable(tf.constant(1.0, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
  b5 = tf.Variable(tf.constant(1.0, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
  b6 = tf.Variable(tf.constant(0.0, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b6')
  b7 = tf.Variable(tf.constant(0.0, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b7')
  b8 = tf.Variable(tf.constant(0.0, shape=[K], dtype=tf.float32), trainable=True, name='b8')

  #b1 = tf.Variable(tf.ones([NUM_FILTER_1])/10)
  #b2 = tf.Variable(tf.ones([NUM_FILTER_2])/10)
  #b3 = tf.Variable(tf.ones([NUM_FILTER_3])/10)
  #b4 = tf.Variable(tf.ones([NUM_FILTER_4])/10)
  #b5 = tf.Variable(tf.ones([NUM_FILTER_5])/10)
  #b6 = tf.Variable(tf.ones([NUM_NEURON_1])/10)
  #b7 = tf.Variable(tf.ones([NUM_NEURON_2])/10)
  #b8 = tf.Variable(tf.ones([K])/10)


  #===== architecture =====#
  conv1 = tf.nn.relu(tf.nn.conv2d(X,  W1, strides=[1,4,4,1], padding='SAME')+b1)
  pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
  norm1 = tf.nn.lrn(pool1, alpha=2e-5, beta=0.75, depth_radius=2, bias=1.0)

  conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
  norm2 = tf.nn.lrn(pool2, alpha=2e-5, beta=0.75, depth_radius=2, bias=1.0)

  conv3 = tf.nn.relu(tf.nn.conv2d(norm2, W3, strides=[1,1,1,1], padding='SAME')+b3)
  conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
  conv5 = tf.nn.relu(tf.nn.conv2d(conv4, W5, strides=[1,1,1,1], padding='SAME')+b5)
  pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

  YY = tf.reshape(pool5, shape=[-1,6*6*NUM_FILTER_5])

  fc1 = tf.nn.relu(tf.matmul(YY,W6)+b6)
  fc1_drop = tf.nn.dropout(fc1, keep_prob_1)

  fc2 = tf.nn.relu(tf.matmul(fc1_drop,W7)+b7)
  fc2_drop = tf.nn.dropout(fc2, keep_prob_1)

  fc3 = tf.matmul(fc2_drop,W8)+b8
  Y   = tf.nn.softmax(fc3)



  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=fc3))
  l2_loss = tf.reduce_sum(LMBDA * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
  total_loss = cross_entropy + l2_loss


  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
  #tf.add_to_collection('losses', cross_entropy)
  #total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')



  global_step = tf.Variable(0, trainable=False)
  #starter_learning_rate = LEARNING_RATE
  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
  #                                           1000000, 0.9, staircase=True)
  #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_nesterov=True).minimize(total_loss)
  train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(total_loss, global_step=global_step)
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


  correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





  # Passing global_step to minimize() will increment it at each step.
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
 

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver() 

  #learning_rate = tf.placeholder(tf.float32, shape=[])
  #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


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

    # Restore variables from disk.
    #saver.restore(sess, "./checkpoint/model_3000.ckpt")
    #print("Model restored.")


    x, y = batchRead(class_name, pool)
    wnid_labels, _ = tu.load_imagenet_meta('/home/hhwu/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat')
    for itr in xrange(1000000):
      #x, y = batchRead(image_name, class_dict, mean_img, pool)


      #print y
      asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7, asyn_train_y = setAsynBatchRead(class_name, pool)
      #start_time = time.time()
      #x, y = tu.read_batch(mini_batch, "/home/hhwu/ImageNet/train/", wnid_labels)
      _, step = sess.run([train_step, global_step], feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1, keep_prob_2: DROPOUT_PROB_2})
      #x, y = batchRead(class_name, pool)
      #elapsed_time = time.time() - start_time
      #print "Time for async read and training: %f" % elapsed_time
      x = getAsynBatchRead(asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7)
      y = asyn_train_y
      

      #print train_step
 
      #print "W9:"
      #print sess.run(W9) 
      if itr % 10 == 0:
        print "Iter %d:  learning rate: %f  dropout: (%.1f %.1f) cross entropy: %f total loss: %f  accuracy: %f" % (itr,
                                                                LEARNING_RATE, 
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


      if itr % 1000 == 0 and itr != 0:
        model_name = "./checkpoint/model_%d.ckpt" % itr
        save_path = saver.save(sess, model_name)
        #save_path = saver.save(sess, "./checkpoint/model.ckpt")
        print("Model saved in file: %s" % save_path)


