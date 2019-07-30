import argparse
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import subprocess
import tensorflow as tf
import time

import model
import util

# configs
parser = argparse.ArgumentParser(description = "Train the Incremental GAN on CelebA data")
parser.add_argument("--hole_size", default = 48, help = "Size of each mask hole")
parser.add_argument("--hole_num", default = 5, help = "Number of mask holes")
parser.add_argument("--epoch", default = 1, help = "Epoch to be evaluated")
parser.add_argument("--gpu", default = -1, help = "GPU ID")
parser.add_argument("--test_num", default = 20, help = "Number of test images")
parser.add_argument("--checkpoint_dir", default = "celeba_gan_incr_checkpoints/", help = "Directory where checkpoints are stored, followed by /")
parser.add_argument("--output_dir", default = "test_outputs/", help = "Directory to store outputs, followed by /")

args = parser.parse_args()

input_height = 216
input_width = 176

hole_size = int(args.hole_size)
hole_num = int(args.hole_num)

test_epoch = int(args.epoch)
test_imgs = int(args.test_num)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

checkpoint_dir = str(args.checkpoint_dir)
output_dir = str(args.output_dir)

# load files
_, test_files = util.get_celeba_paths()
image_loader = util.load_celeba_image

random.seed(0)

# model
inpainter = model.Inpainter(input_height, input_width)

def clear_folders():
	for file in os.scandir(output_dir + "real/"):
		if file.name.endswith(".png"):
			os.unlink(file.path)

	for file in os.scandir(output_dir + "inpaint_1/"):
		if file.name.endswith(".png"):
			os.unlink(file.path)

	for file in os.scandir(output_dir + "inpaint_2/"):
		if file.name.endswith(".png"):
			os.unlink(file.path)

# prepare
util.mkdir_if_needed(output_dir)
util.mkdir_if_needed(output_dir + "real")
util.mkdir_if_needed(output_dir + "inpaint_1")
util.mkdir_if_needed(output_dir + "inpaint_2")
util.mkdir_if_needed(output_dir + "combine")

# core
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(max_to_keep = None)

	saver.restore(sess, checkpoint_dir + "epoch%02d" % test_epoch)

	test_idxs = random.sample(list(range(len(test_files))), test_imgs)

	inpaint1_l2s = []
	inpaint2_l2s = []
	inpaint1_fids = []
	inpaint2_fids = []

	k = 1

	for test_idx in test_idxs:
		real_images_batch = []
		masks_batch = []

		clear_folders()

		image_array = image_loader(test_files[test_idx])
		cv2.imwrite(output_dir + "%d_real.png" % (test_idx), image_array * 255.)
		cv2.imwrite(output_dir + "real/%d.png" % (test_idx), image_array * 255.)

		mask = util.generate_box_mask(width = input_width, height = input_height, hole_size = hole_size, num_holes = hole_num)

		real_images_batch.append(image_array)
		masks_batch.append(mask)

		cv2.imwrite(output_dir + "%d_masked.png" % (test_idx), np.multiply(image_array, mask) * 255.)

		real_images_batch = np.array(real_images_batch)
		masks_batch = np.array(masks_batch)

		outputs = sess.run([inpainter.inpaint1, inpainter.inpaint2, inpainter.confidence1, inpainter.confidence2], feed_dict = {
			inpainter.real_images: real_images_batch,
			inpainter.masks: masks_batch
		})

		for i in range(len(outputs)):
			if i < 2:
				outputs[i] = np.reshape(outputs[i], (input_height, input_width, 3))
			else:
				outputs[i] = np.reshape(outputs[i], (input_height, input_width, 1))

		cv2.imwrite(output_dir + "%d_inpaint_1.png" % (test_idx), outputs[0] * 255.)
		cv2.imwrite(output_dir + "inpaint_1/%d.png" % (test_idx), outputs[0] * 255.)
		cv2.imwrite(output_dir + "%d_inpaint_2.png" % (test_idx), outputs[1] * 255.)
		cv2.imwrite(output_dir + "inpaint_2/%d.png" % (test_idx), outputs[1] * 255.)
		cv2.imwrite(output_dir + "%d_conf_1.png" % (test_idx), outputs[2] * 255.)
		cv2.imwrite(output_dir + "%d_conf_2.png" % (test_idx), outputs[3] * 255.)

		combined = np.concatenate([image_array, np.multiply(image_array, mask), outputs[0], outputs[1], np.repeat(outputs[2], 3, axis = 2), np.repeat(outputs[3], 3, axis = 2)], axis = 1)
		cv2.imwrite(output_dir + "combine/%d.png" % (test_idx), combined * 255.)

		inpaint1_l2 = np.mean(np.square(outputs[0] - image_array))
		inpaint2_l2 = np.mean(np.square(outputs[1] - image_array))

		inpaint1_l2s.append(inpaint1_l2)
		inpaint2_l2s.append(inpaint2_l2)

		cmd = ["python", "fid_score.py", output_dir + "real/", output_dir + "inpaint_1/"]
		subprocess.Popen(cmd).wait()

		with open("fid_score.txt", "r") as fp:
			inpaint1_fid = float(fp.read())

		cmd = ["python", "fid_score.py", output_dir + "real/", output_dir + "inpaint_2/"]
		subprocess.Popen(cmd).wait()

		with open("fid_score.txt", "r") as fp:
			inpaint2_fid = float(fp.read())

		inpaint1_fids.append(inpaint1_fid)
		inpaint2_fids.append(inpaint2_fid)

		print("Epoch %d, sample %d, Inpaint 1 L2 dist = %.3e, inpaint 2 L2 dist = %.3e" % (test_epoch, k, inpaint1_l2, inpaint2_l2))
		print("Epoch %d, sample %d, Inpaint 1 FID dist = %.3f, inpaint 2 FID dist = %.3f" % (test_epoch, k, inpaint1_fid, inpaint2_fid))

		k += 1

print("Inpaint 1 L2 avg = %.3e, inpaint 2 L2 avg = %.3e" % (np.mean(inpaint1_l2s), np.mean(inpaint2_l2s)))
print("Inpaint 1 FID avg = %.3e, inpaint 2 FID avg = %.3e" % (np.mean(inpaint1_fids), np.mean(inpaint2_fids)))

with open("dist_output_epoch%02d.json" % test_epoch, "w") as fp:
	json.dump({
		"inpaint1_l2": inpaint1_l2s,
		"inpaint2_l2": inpaint2_l2s,
		"inpaint1_fid": inpaint1_fids,
		"inpaint2_fid": inpaint2_fids
	}, fp, indent = 4)