#!/usr/bin/python

import tensorflow as tf
import numpy as np
import sys
import csv
import os
from skimage import io
from multiprocessing.pool import ThreadPool


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

  class_dict = {}
  for i in xrange(0,len(class_name)):
    class_dict[class_name[i]] = i


  return class_dict


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
  datapath = '/home1/hhwu/ILSVRC2012_img_val/'
  print "Path: ", datapath

  image_name = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    #print "The number of files: %d" % len(filenames)
    
    image_name = filenames


  print "The number of files: %d" % len(image_name)

  out = open("/home1/hhwu/data/ILSVRC2012_validation_ground_truth.txt", 'r')
  lines = out.readlines()

  print "number of labels: ", len(lines)

  out.close()
 

#  data_list = []
#  for i in idx:
#    absfile = os.path.join(dirpath, image_name[i]) 
#    print image_name[i]
#    print absfile
#    data_list.append(absfile)
#
#  output_name = "train.bin" 
#  ouf = open(output_name, 'w')
#  cPickle.dump(data_list, ouf, 1)
#  print "File %s is written." % output_name
 
  image_counter = 0
  data_list  = []
  label_list = []
  data_dict = {}
  output_name = "valid.tfrecords"
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in xrange(0,len(lines)):
    absfile = os.path.join(dirpath, image_name[i]) 
    target_img = io.imread(absfile)
 
    feature = {'valid/label': _int64_feature(int(lines[i])),
               'valid/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  write.close()

    #print class_name[image_name[idx[i]].split('_')[0]]
    #print image_name[idx[i]].split('_')[0]
#
#  #output_name = "train_%d.bin" % image_counter
#  #ouf = open(output_name, 'w')
#  #cPickle.dump(data_dict, ouf, 1)
#  #print "File %s is written." % output_name
#
#  output_name = "train_%d.tfrecords" % image_counter
#  writer = tf.python_io.TFRecordWriter(output_name)
#  data_list
#  label_list
#
#  for j in xrange(0, len(label_list)):
#    feature = {'train/label': _int64_feature(label_list[j]),
#               'train/image': _bytes_feature(tf.compat.as_bytes(data_list[j].tostring()))}
#    # Create an example protocol buffer
#    example = tf.train.Example(features=tf.train.Features(feature=feature))
#  
#    # Serialize to string and write on the file
#    writer.write(example.SerializeToString())
#
#  print "File %s is written." % output_name
 


#  data_list = []
#  for i in range(0,3000000):
#  #for i in range(0,I.shape[0]):
#    IQ_pair = np.zeros(2)
#    IQ_pair[0] = I[i]
#    IQ_pair[1] = Q[i]
#
#    data_list.append(IQ_pair)
#
#  ouf = open(sys.argv[1], 'w')
#  cPickle.dump(data_list, ouf, 1)
#  ouf.close()
#  #ouf = open(sys.argv[2], 'w')
#  #cPickle.dump(Q[0:15000], ouf, 1)
#  #ouf.close()
#
#  fo = open(sys.argv[1], 'rb')
#  test_file = cPickle.load(fo)
#  fo.close()
#
#  #print test_file
#  print len(test_file)
