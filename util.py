import cv2
import numpy as np
import os
import random
import tensorflow as tf

def generate_box_mask(width = 216, height = 176, hole_size = 16, num_holes = 8):
	mask = np.ones((height, width, 1))

	for _ in range(num_holes):
		top = random.randint(0, height - hole_size - 1)
		left = random.randint(0, width - hole_size - 1)

		bottom = top + hole_size
		right = left + hole_size

		mask[top:bottom, left:right, :] = 0.

	return mask

def generate_pepper_mask(width = 216, height = 176, p = 0.7):
	return np.random.choice([0., 1.], size=(height, width, 1), p=[p, 1 - p])

def get_celeba_paths():
	train_files = []
	test_files = []

	with open("data/list_eval_partition.csv", "r") as fp:
		for idx, line in enumerate(fp):
			if idx != 0:
				partition = int(line.split(",")[1])

				if partition == 0 or partition == 1:
					train_files.append("data/img_align_celeba/" + line.split(",")[0])
				elif partition == 2:
					test_files.append("data/img_align_celeba/" + line.split(",")[0])

	return train_files, test_files

def get_coco_paths():
	train_files = []
	test_files = []

	train_files_name = os.listdir("coco/images/train2017/")
	val_files_name = os.listdir("coco/images/val2017/")
	test_files_name = os.listdir("coco/images/test2017/")

	for train_file_name in train_files_name:
		train_files.append("coco/images/train2017/" + train_file_name)

	for val_file_name in val_files_name:
		train_files.append("coco/images/val2017/" + val_file_name)

	for test_file_name in test_files_name:
		test_files.append("coco/images/test2017/" + test_file_name)

	return train_files, test_files

def load_celeba_image(filename):
	image_array = cv2.imread(filename).astype("float")
	image_array /= 255.
	image_array = image_array[1:-1, 1:-1, :]

	return image_array

def load_image_crop(filename, resize_size = 384, crop_size = 128):
	img = cv2.imread(filename).astype("float")
	img /= 255.
	height, width, _ = img.shape

	if height < width:
		new_height = resize_size
		new_width = int(float(width) / float(height) * resize_size)
	else:
		new_width = resize_size
		new_height = int(float(height) / float(width) * resize_size)

	img = cv2.resize(img, (new_width, new_height))

	left = random.randint(0, new_width - crop_size)
	top = random.randint(0, new_height - crop_size)
	right = left + crop_size
	bottom = top + crop_size

	return img[top:bottom, left:right, :]

def soft_zeros_like(tensor):
	return tf.zeros_like(tensor) + tf.random_uniform(tf.shape(tensor), minval = 0, maxval = 0.2)

def soft_ones_like(tensor):
	return tf.ones_like(tensor) + tf.random_uniform(tf.shape(tensor), minval = -0.2, maxval = 0.2)

def soft_labels(tensor):
	return tf.maximum(tensor + tf.random_uniform(tf.shape(tensor), minval = -0.2, maxval = 0.2), 0)

def mkdir_if_needed(dir_name):
	if dir_name[-1] == "/":
		dir_name = dir_name[:-1]

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)