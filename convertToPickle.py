#!/usr/bin/python

import cPickle
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



if __name__ == '__main__':
  datapath = '/home/hhwu/ImageNet/crop_train/'
  num_images = 1281167
  file_size = 25600
  print "Path: ", datapath

  class_dict  = loadClassName('synset.csv')


  image_name = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    #print "The number of files: %d" % len(filenames)
    
    image_name = filenames


  idx = np.random.randint(0,num_images,num_images)
  print "The number of files: %d" % len(image_name)

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
  for i in xrange(0,len(idx)):
    if i % file_size == 0:
      if i == 0:
        data_list  = []
        label_list = []
      else:
        output_name = "train_%d.bin" % image_counter
        ouf = open(output_name, 'w')
        data_dict = {}
        data_dict['data'] = data_list
        data_dict['label'] = label_list
        cPickle.dump(data_list, ouf, 1)
        print "File %s is written." % output_name
        data_list = []
        label_list = []
        image_counter += 1
        print i
    

    absfile = os.path.join(dirpath, image_name[idx[i]]) 
    target_img = io.imread(absfile)
    data_list.append(target_img)
    label_list.append(int(class_dict[image_name[idx[i]].split("_")[0]]))
    #print class_name[image_name[idx[i]].split('_')[0]]
    #print image_name[idx[i]].split('_')[0]

  output_name = "train_%d.bin" % image_counter
  ouf = open(output_name, 'w')
  cPickle.dump(data_list, ouf, 1)
  print "File %s is written." % output_name




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
