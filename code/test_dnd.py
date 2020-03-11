from __future__ import division
from __future__ import print_function
import os
import argparse
import tensorflow as tf
import numpy as np
import imageio
import h5py
import scipy.io as sio
from model import *

parser = argparse.ArgumentParser(description='Testing on DND dataset')
parser.add_argument('--ckpt', type=str, default='SIDD_transfer',
                    choices=['SIDD_transfer', 'AINDNet'], help='checkpoint type')
parser.add_argument('--input_dir', type=str, default='/data2/terryoo/DB/', help='input directory of DND mat file')
parser.add_argument('--bd_margin', dest='bd_margin', type=int, default=80, help='borderline_margin')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='GPU ID')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
max_margin = args.bd_margin
data_folder = args.input_dir
checkpoint_dir = './checkpoint/' + args.ckpt
model_name = 'model'

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx,yy,bb]
    return sigma

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma

# model setting
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
_, out_image = AINDNet(in_image)

# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
model_path = checkpoint_dir + '/' + model_name
if ckpt:
    print('loaded', model_path)
    print(model_path)
    saver.restore(sess, model_path)

result_dir = './results/mat/' + args.ckpt + '_' + model_name + '/'
out_folder_images ='./results/images/' + args.ckpt + '_' + model_name + '/'

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(out_folder_images):
    os.makedirs(out_folder_images)

infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']
print('info loaded\n')
for i in range(50):
    filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
    img = h5py.File(filename, 'r')
    Inoisy = np.float32(np.array(img['InoisySRGB']).T)
    # bounding box
    ref = bb[0][i]
    boxes = np.array(info[ref]).T
    for k in range(20):
        idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]

        # Crop margin for better boundary process
        h_min_margin = max_margin
        h_max_margin = max_margin
        w_min_margin = max_margin
        w_max_margin = max_margin

        if 0 > idx[0] - max_margin:
            h_min_margin = idx[0]
        if Inoisy.shape[0] < idx[1] + max_margin:
            h_max_margin = Inoisy.shape[0] - idx[1]
        if 0 > idx[2] - max_margin:
            w_min_margin = idx[2]
        if Inoisy.shape[1] < idx[3] + max_margin:
            w_max_margin = Inoisy.shape[1] - idx[3]

        Inoisy_crop = Inoisy[idx[0] - h_min_margin:idx[1] + h_max_margin,
                      idx[2] - w_min_margin:idx[3] + w_max_margin, :].copy()
        H = Inoisy_crop.shape[0]
        W = Inoisy_crop.shape[1]


        nlf = load_nlf(info, i)
        nlf["sigma"] = load_sigma_srgb(info, i, k)
        Idenoised_crop = sess.run(out_image, feed_dict={in_image: Inoisy_crop[None, :, :, :]})
        Idenoised_crop2 = Idenoised_crop[:, h_min_margin:-h_max_margin, w_min_margin:-w_max_margin, :]
        # save denoised data
        Idenoised_crop2 = np.float32(Idenoised_crop2)
        save_file = os.path.join(result_dir, '%04d_%02d.mat' % (i + 1, k + 1))
        sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop2})

        # save denoised image
        save_file_dnoisied = os.path.join(out_folder_images, '%04d_%02d_denoised.png' % (i + 1, k + 1))
        Idenoised_crop_image = np.clip(255 * Idenoised_crop2, 0, 255).astype('uint8')
        imageio.imwrite(save_file_dnoisied, Idenoised_crop_image[0,:,:,:])
        print('%s crop %d/%d' % (filename, k + 1, 20))
    print('[%d/%d] %s done\n' % (i + 1, 50, filename))


