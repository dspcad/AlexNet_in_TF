#!/usr/bin/python

import tensorflow as tf
import numpy as np
import sys
import csv
import os
from skimage import io
from multiprocessing.pool import ThreadPool



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
  datapath = '/home1/hhwu/ImageNet/crop_valid/'
  print "Path: ", datapath

  image_name = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    #print "The number of files: %d" % len(filenames)
    
    image_name = filenames

  image_name.sort()
  #print image_name
  #print "The number of files: %d" % len(image_name)

  out = open("/home1/hhwu/data/ILSVRC2012_validation_ground_truth.txt", 'r')
  lines = out.readlines()

  print "number of labels: ", len(lines)

  out.close()
 
  image_counter = 0
  data_list  = []
  label_list = []
  data_dict = {}
  output_name = "valid.tfrecords"
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in xrange(0,len(lines)):
    absfile = os.path.join(dirpath, image_name[i]) 
    target_img = io.imread(absfile)
 
    print "%d image." % i
    feature = {'valid/label': _int64_feature(int(lines[i])-1),
               'valid/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  writer.close()

