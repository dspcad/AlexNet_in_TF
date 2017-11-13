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
      #initializer=tf.contrib.layers.xavier_initializer()
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
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


def batchCroppedImgRead(thread_name, dirpath, class_name, mean_img, partial_batch_idx):
  #print "%s is cropping the images..." % thread_name
  img_batch = []

  for i in partial_batch_idx:
    #absfile = os.path.join(dirpath, image_name[i])
    #target_img = io.imread(absfile)

    images_folder = os.path.join(dirpath, class_name[i])
    absfile = os.path.join(images_folder, np.random.choice(os.listdir(images_folder)))
    target_img = io.imread(absfile)

    #################################
    # convert RGB from float to int #
    #################################
    croppedImg = cropImg(target_img, mean_img)

    if len(img_batch) == 0:
      img_batch = croppedImg
    else:
      img_batch = np.vstack((img_batch, croppedImg))


  return img_batch




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

  #np.random.seed(31)


  class_name  = loadClassName('synset.csv')

  pool = ThreadPool(processes=8)
  print "Multi-threads begin!"

  
  

  
  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 256

  K = 1000 # number of classes
  NUM_FILTER_1 = 12
  NUM_FILTER_2 = 128

  DROPOUT_PROB = 0.50

  LEARNING_RATE = 1e-2
  NUM_IMAGES = 1281167  

  reg = 0 # regularization strength


  # Dropout probability
  keep_prob     = tf.placeholder(tf.float32)
  #learning_rate = tf.placeholder(tf.float32)

  # initialize parameters randomly

  X  = tf.placeholder(tf.float32, shape=[None, 227,227,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1_1  = _variable_with_weight_decay('W1_1', shape=[11, 11, 3, NUM_FILTER_1], stddev=1e-2, wd=5e-4)
  W1_2  = _variable_with_weight_decay('W1_2', shape=[11, 11, 3, NUM_FILTER_1], stddev=1e-2, wd=5e-4)

  W2_1  = _variable_with_weight_decay('W2_1', shape=[5, 5, NUM_FILTER_1,NUM_FILTER_2], stddev=1e-2, wd=5e-4)
  W2_2  = _variable_with_weight_decay('W2_2', shape=[5, 5, NUM_FILTER_1,NUM_FILTER_2], stddev=1e-2, wd=5e-4)

  W3    = _variable_with_weight_decay('W8', shape=[13*13*NUM_FILTER_2*2,K], stddev=1e-2, wd=5e-4)
  

  b1_1 = tf.Variable(tf.constant(0.0, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1_1')
  b1_2 = tf.Variable(tf.constant(0.0, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1_2')

  b2_1 = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2_1')
  b2_2 = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2_2')

  b3   = tf.Variable(tf.constant(0.0, shape=[K], dtype=tf.float32), trainable=True, name='b8')




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


  YY = tf.reshape(cross2, shape=[-1,13*13*NUM_FILTER_2*2])

  #===== Layer 3 =====#
  Y = tf.matmul(YY,W3)+b3             



  #for var in tf.trainable_variables():
  #  print var


  #diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
  tf.add_to_collection('losses', cross_entropy)
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
 

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                             100000, 0.1, staircase=True)

  train_step = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(total_loss)
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
  #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
  correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.uint8))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  # Passing global_step to minimize() will increment it at each step.
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
 

  #####################################################
  #        Save the W1 and W2 parameters only         #
  #####################################################
  saver = tf.train.Saver([W1_1,W1_2,W2_1,W2_2])


  # Add ops to save and restore all the variables.
  #saver = tf.train.Saver() 

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

  valid_data_path = "/mnt/ramdisk/valid.tfrecords"
  #valid_data_path = "/home1/hhwu/ImageNet/valid.tfrecords"
  train_data_path = []
  for i in xrange(0,101):
    train_data_path.append("/mnt/ramdisk/tf_data/train_%d.tfrecords" % i)
    #train_data_path.append("/home1/hhwu/ImageNet/tf_data/train_%d.tfrecords" % i)


  with tf.Session() as sess:
    ################################
    #        Training Data         #
    ################################
    train_feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    train_filename_queue = tf.train.string_input_producer(train_data_path, num_epochs=90)
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    train_reader = tf.TFRecordReader()
    _, train_serialized_example = train_reader.read(train_filename_queue)

        # Decode the record read by the reader
    train_features = tf.parse_single_example(train_serialized_example, features=train_feature)
    # Convert the image data from string back to the numbers
    train_image = tf.cast(tf.decode_raw(train_features['train/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    train_label_idx = tf.cast(train_features['train/label'], tf.int32)
    train_label = tf.one_hot(train_label_idx, K)
    # Reshape image data into the original shape
    train_image = tf.reshape(train_image, [256, 256, 3])

    train_image = cropImg(train_image, mean_img)

    train_images, train_labels = tf.train.shuffle_batch([train_image, train_label], 
                                                         batch_size=mini_batch, capacity=3*mini_batch, num_threads=16, min_after_dequeue=256)


    ################################
    #       Validation Data        #
    ################################
    valid_feature = {'valid/image': tf.FixedLenFeature([], tf.string),
                     'valid/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    valid_filename_queue = tf.train.string_input_producer([valid_data_path], num_epochs=500)
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    valid_reader = tf.TFRecordReader()
    _, valid_serialized_example = valid_reader.read(valid_filename_queue)

        # Decode the record read by the reader
    valid_features = tf.parse_single_example(valid_serialized_example, features=valid_feature)
    # Convert the image data from string back to the numbers
    valid_image = tf.cast(tf.decode_raw(valid_features['valid/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    valid_label_idx = tf.cast(valid_features['valid/label'], tf.int32)
    valid_label = tf.one_hot(valid_label_idx, K)
    # Reshape image data into the original shape
    valid_image = tf.reshape(valid_image, [256, 256, 3])

    valid_image = cropImg(valid_image, mean_img)

    valid_images, valid_labels = tf.train.batch([valid_image, valid_label], 
                                                         batch_size=1000, capacity=50000, num_threads=16)




    #sess.run(tf.global_variables_initializer())
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)


    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #x, y = batchRead(class_name, mean_img, pool)

    #test_x = []
    #test_y = []
    #for i in range(0,50):
    #  if i == 0:
    #    test_x, test_y = sess.run([valid_images, valid_labels])
    #  else:
    #    tmp_x, tmp_y = sess.run([valid_images, valid_labels])
    #    test_x = np.vstack((test_x, tmp_x))
    #    test_y = np.vstack((test_y, tmp_y))

  
    #test_x = test_x.reshape(50000,227,227,3)
    #test_y = test_y.reshape(50000,K)


    image_iterator = 0
    data = []
    label = []
    for itr in xrange(100000):
      #x, y = batchRead(image_name, class_dict, mean_img, pool)

      #print y
      #asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7, asyn_train_y = setAsynBatchRead(class_name, pool, mean_img)
      #start_time = time.time()

      #x, y, image_iterator, data, label = batchSerialRead(image_iterator, data, label)
      x, y = sess.run([train_images, train_labels])
      train_step.run(feed_dict={X: x, Y_: y, keep_prob: DROPOUT_PROB})
      #elapsed_time = time.time() - start_time
      #print "Time for training: %f" % elapsed_time
      if itr % 20 == 0:
        print "Iter %d:  learning rate: %f  dropout: %.1f cross entropy: %f total loss: %f  accuracy: %f" % (itr,
                                                                learning_rate.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}),
                                                                DROPOUT_PROB,
                                                                cross_entropy.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}),
                                                                total_loss.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}),
                                                                accuracy.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}))

      if itr % 500 == 0 and itr != 0:
        valid_accuracy = 0.0
        for i in range(0,50):
          test_x, test_y = sess.run([valid_images, valid_labels])
          valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, keep_prob: 1.0})
        print "Validation Accuracy: %f (%.1f/50000)" %  (valid_accuracy/50000, valid_accuracy)
       


      if epoch_counter*mini_batch > NUM_IMAGES:
        epoch_counter = 0
        epoch_num = epoch_num + 1
        print "Epoch: ", epoch_num
        valid_accuracy = 0.0
        for i in range(0,50):
          test_x, test_y = sess.run([valid_images, valid_labels])
          valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, keep_prob: 1.0})
        print "Validation Accuracy: %f (%.1f/50000)" %  (valid_accuracy/50000, valid_accuracy)
 
      else:
        epoch_counter = epoch_counter + 1


    coord.request_stop()
    coord.join(threads)
    sess.close()

    #if itr % 10000 == 0 and itr != 0:
    model_name = "./checkpoint/model_small_1.ckpt"
    save_path = saver.save(sess, model_name)
    #save_path = saver.save(sess, "./checkpoint/model.ckpt")
    print("Model saved in file: %s" % save_path)




