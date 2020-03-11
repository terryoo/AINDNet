from __future__ import division
from __future__ import print_function
import os
import argparse
import tensorflow as tf
import numpy as np
import imageio
from glob import glob
from model import *

parser = argparse.ArgumentParser(description='Testing on image dataset')
parser.add_argument('--ckpt', type=str, default='SIDD_transfer',
                    choices=['SIDD_transfer', 'AINDNet'], help='checkpoint type')
parser.add_argument('--input_dir', type=str, default='./testset/RNI15', help='evaluate path')
parser.add_argument('--output_dir', type=str, default='./results/RNI15/', help='output path')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='GPU ID')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
data_folder = args.input_dir
checkpoint_dir = './checkpoint/' + args.ckpt
model_name = 'model'


# model setting
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
_, out_image = AINDNet(in_image)

# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
model_path = checkpoint_dir + '/' + model_name
print('loaded', model_path)
print(model_path)
saver.restore(sess, model_path)

out_folder_images = args.output_dir
full_path = args.input_dir + '/*'
test_data= glob(full_path)

if not os.path.isdir(out_folder_images):
    os.makedirs(out_folder_images)

for img_idx in range(len(test_data)):
    noisy_image = imageio.imread(test_data[img_idx])/255.0
    filename = test_data[img_idx].split('/')[-1]
    output_clean_image = sess.run(out_image, feed_dict={in_image: noisy_image[None,:,:,:]})
    outputimage = np.clip(np.round(255 * output_clean_image[0]), 0, 255).astype('uint8')
    imageio.imwrite(out_folder_images + filename[:-4] + '.png', outputimage)


