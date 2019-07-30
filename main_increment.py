import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

import model
import util

# configs
parser = argparse.ArgumentParser(description = "Train the Incremental GAN on CelebA data")
parser.add_argument("--hole_size", default = 48, help = "Size of each mask hole")
parser.add_argument("--hole_num", default = 5, help = "Number of mask holes")
parser.add_argument("--batch_size", default = 16, help = "Batch size")
parser.add_argument("--gpu", default = -1, help = "GPU ID")
parser.add_argument("--checkpoint_dir", default = "celeba_gan_incr_checkpoints/", help = "Directory to store checkpoints, followed by /")
parser.add_argument("--output_dir", default = "celeba_gan_incr_outputs/", help = "Directory to store outputs, followed by /")
parser.add_argument("--log_path", default = "celeba_log_incr.txt", help = "Log file path")

args = parser.parse_args()

input_height = 216
input_width = 176

hole_size = int(args.hole_size)
hole_num = int(args.hole_num)

batch_size = int(args.batch_size)

checkpoint_dir = str(args.checkpoint_dir)
output_dir = str(args.output_dir)

log_path = str(args.log_path)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# load files
train_files, test_files = util.get_celeba_paths()
image_loader = util.load_celeba_image
		
tf.reset_default_graph()
tf.set_random_seed(0)
random.seed(0)

# model
inpainter = model.Inpainter(input_height, input_width)

fpLog = open(log_path, "w")

# prepare
util.mkdir_if_needed(checkpoint_dir)
util.mkdir_if_needed(output_dir)

# core
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(max_to_keep = None)

	for epoch in range(10):
		random.shuffle(train_files)

		print("Epoch %d" % (epoch + 1))
		fpLog.write("Epoch %d\n" % (epoch + 1))
		fpLog.flush()

		tic = time.clock()

		# training
		for i in range(0, 2048, batch_size):
			start_idx = i
			end_idx = min(i + batch_size, len(train_files))

			real_images_batch = []
			masks_batch = []

			for j in range(start_idx, end_idx):
				real_images_batch.append(image_loader(train_files[j]))
				masks_batch.append(util.generate_box_mask(width = input_width, height = input_height, hole_size = hole_size, num_holes = hole_num))

			real_images_batch = np.array(real_images_batch)
			masks_batch = np.array(masks_batch)

			D1_loss_batch, _, D2_loss_batch, _ = sess.run([inpainter.D1_loss_mean, inpainter.D1_trainer, inpainter.D2_loss_mean, inpainter.D2_trainer], feed_dict = {
				inpainter.real_images: real_images_batch,
				inpainter.masks: masks_batch
			})

			G1_loss_batch, _, G2_loss_batch, _ = sess.run([inpainter.G1_loss_mean, inpainter.G1_trainer, inpainter.G2_loss_mean, inpainter.G2_trainer], feed_dict = {
				inpainter.real_images: real_images_batch,
				inpainter.masks: masks_batch
			})

			if i % 512 == 0:
				fpLog.write("Epoch %d, sample %d, D1_loss = %lf, D2_loss = %lf, G1_loss = %lf, G2_loss = %lf\n" % (epoch + 1, i, D1_loss_batch, D2_loss_batch, G1_loss_batch, G2_loss_batch))
				fpLog.flush()

		saver.save(sess = sess, save_path = checkpoint_dir + "epoch%02d" % (epoch + 1))

		toc = time.clock()

		fpLog.write("Training time = %lf\n" % (toc - tic))
		fpLog.flush()

		util.mkdir_if_needed(output_dir + "epoch%d" % (epoch + 1))

		# testing
		test_idxs = random.sample(list(range(len(test_files))), 10)

		for test_idx in test_idxs:
			real_images_batch = []
			masks_batch = []

			image_array = image_loader(test_files[test_idx])

			cv2.imwrite(output_dir + "epoch%d/%d_real.png" % (epoch + 1, test_idx), image_array * 255.)

			mask = util.generate_box_mask(width = input_width, height = input_height, hole_size = hole_size, num_holes = hole_num)

			real_images_batch.append(image_array)
			masks_batch.append(mask)

			cv2.imwrite(output_dir + "epoch%d/%d_masked.png" % (epoch + 1, test_idx), np.multiply(image_array, mask) * 255.)

			real_images_batch = np.array(real_images_batch)
			masks_batch = np.array(masks_batch)

			outputs = sess.run([inpainter.inpaint1, inpainter.inpaint2, inpainter.confidence1, inpainter.confidence2], feed_dict = {
				inpainter.real_images: real_images_batch,
				inpainter.masks: masks_batch
			})

			for i in range(len(outputs)):
				if outputs[i].size == input_height * input_width * 3:
					outputs[i] = np.reshape(outputs[i], (input_height, input_width, 3))
				elif outputs[i].size == input_height * input_width:
					outputs[i] = np.reshape(outputs[i], (input_height, input_width, 1))

			cv2.imwrite(output_dir + "epoch%d/%d_inpaint_1.png" % (epoch + 1, test_idx), outputs[0] * 255.)
			cv2.imwrite(output_dir + "epoch%d/%d_inpaint_2.png" % (epoch + 1, test_idx), outputs[1] * 255.)
			cv2.imwrite(output_dir + "epoch%d/%d_conf_1.png" % (epoch + 1, test_idx), outputs[2] * 255.)
			cv2.imwrite(output_dir + "epoch%d/%d_conf_2.png" % (epoch + 1, test_idx), outputs[3] * 255.)

fpLog.close()