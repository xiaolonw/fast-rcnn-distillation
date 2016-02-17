
import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np


caffe.set_mode_cpu()
net = caffe.Net('/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/scripts/pre_gan2/train.prototxt', caffe.TRAIN)
headfile = '/nfs.yoda/xiaolonw/torch_projects/weights4/head_r.txt'
f = open(headfile, 'r')

savename = '/nfs.yoda/xiaolonw/fast_rcnn/models/pre_gan2/fast_rcnn_zero.caffemodel'

layer_num = 5
layernames = ('da_conv1', 'da_conv2', 'da_conv3', 'da_conv4', 'da_conv5')


for i in xrange(layer_num):
	layer_name = layernames[i]
	weight_dims = np.shape(net.params[layer_name][0].data)
	bias_dims   = np.shape(net.params[layer_name][1].data)
	names = f.readline()
	names = names.split(' ')
	lname_weight = names[0] 
	vname_weight = names[1]
	filename_weight = names[2][0:-1] 
	param_num = int(f.readline())
	print(lname_weight) 
	assert(param_num == 4)
	for j in xrange(param_num): 
		now_num = int(f.readline())
		print(now_num)
		print(weight_dims[j])
		assert(now_num == weight_dims[j])

	names = f.readline()
	names = names.split(' ')
	lname_bias = names[0] 
	vname_bias = names[1]
	filename_bias = names[2][0:-1] 
	param_num = int(f.readline())
	assert(param_num == 1)
	now_num = int(f.readline())
	assert(now_num == bias_dims[0])

	f2 = open(filename_weight, 'r')

	for b in xrange(weight_dims[0]):
		for c in xrange(weight_dims[1]):
			for h in xrange(weight_dims[2]):
				for w in xrange(weight_dims[3]):
					net.params[layer_name][0].data[b][c][h][w] = float(f2.readline())

	f2.close()
	# print(net.params[layer_name][0].data[1][2][3][4])

	f3 = open(filename_bias, 'r')

	for j in xrange(bias_dims[0]):
		net.params[layer_name][1].data[j] = float(f3.readline())

	f3.close()


f.close()


net.save(str(savename))









