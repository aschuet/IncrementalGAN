import numpy as np
import tensorflow as tf

import util

def residual_block(x, name = "", depth = 128):
	with tf.variable_scope(name):
		conv1 = tf.keras.layers.Conv2D(filters = depth, kernel_size = 5, use_bias = False, padding = "SAME")(x)
		norm1 = tf.contrib.layers.instance_norm(conv1)
		lrelu1 = tf.nn.leaky_relu(norm1, alpha = 0.1)

		conv2 = tf.keras.layers.Conv2D(filters = depth, kernel_size = 5, dilation_rate = 2, use_bias = False, padding = "SAME")(lrelu1)
		lrelu2 = tf.nn.leaky_relu(conv2, alpha = 0.1)

		conv3 = tf.keras.layers.Conv2D(filters = depth, kernel_size = 5, dilation_rate = 4, use_bias = False, padding = "SAME")(lrelu2)
		lrelu3 = tf.nn.leaky_relu(conv3, alpha = 0.1)

		conv4 = tf.keras.layers.Conv2D(filters = depth, kernel_size = 5, dilation_rate = 8, use_bias = False, padding = "SAME")(lrelu3)
		norm2 = tf.contrib.layers.instance_norm(conv4)
		lrelu4 = tf.nn.leaky_relu(norm2, alpha = 0.1)

		residual = tf.add(x, lrelu4)

	return residual

def increment_generator_first(y, mask, input_width, input_height, prefix = "gen_incr", reuse = None):
	with tf.variable_scope(prefix, reuse = reuse):
		masked_images = tf.multiply(y, mask)

		conv1 = tf.layers.conv2d(masked_images, filters = 32, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		conv2 = tf.layers.conv2d(conv1, filters = 64, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		res1 = residual_block(conv2, name = "generator_mask_res1", depth = 64)
		res2 = residual_block(res1, name = "generator_mask_res2", depth = 64)
		res3 = residual_block(res2, name = "generator_mask_res3", depth = 64)

		conv3 = tf.layers.conv2d(res3, filters = 32, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		conv4 = tf.layers.conv2d(conv3, filters = 3, kernel_size = 3, activation = tf.nn.tanh, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		gen_image = tf.multiply(tf.add(conv4, 1), 0.5)

		output = tf.add(tf.multiply(y, mask), tf.multiply(gen_image, (1 - mask)))

		batch_size = tf.shape(output)[0]
		return output

def increment_generator_feedback(y, mask, conf, input_width, input_height, prefix = "gen_incr_feedback", reuse = None):
	with tf.variable_scope(prefix, reuse = reuse):
		inputs = tf.concat([y, mask, conf], axis = -1)

		conv1 = tf.layers.conv2d(inputs, filters = 32, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		conv2 = tf.layers.conv2d(conv1, filters = 64, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		res1 = residual_block(conv2, name = "generator_mask_res1", depth = 64)
		res2 = residual_block(res1, name = "generator_mask_res2", depth = 64)
		res3 = residual_block(res2, name = "generator_mask_res3", depth = 64)

		conv3 = tf.layers.conv2d(res3, filters = 32, kernel_size = 3, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		conv4 = tf.layers.conv2d(conv3, filters = 3, kernel_size = 3, activation = tf.nn.tanh, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		gen_image = tf.multiply(tf.add(conv4, 1), 0.5)

		output = tf.add(tf.multiply(y, mask), tf.multiply(gen_image, (1 - mask)))

		batch_size = tf.shape(output)[0]
		return output

def increment_discriminator(x, input_width, input_height, prefix = "dis_incr", reuse = None):
	with tf.variable_scope(prefix, reuse = reuse):
		hidden1 = tf.layers.conv2d(x, filters = 32, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		hidden2 = tf.layers.conv2d(hidden1, filters = 64, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		hidden2_pool = tf.keras.layers.AveragePooling2D()(hidden2)

		hidden3 = tf.layers.conv2d(hidden2_pool, filters = 64, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		hidden4 = tf.layers.conv2d(hidden3, filters = 128, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		hidden4_pool = tf.keras.layers.AveragePooling2D()(hidden4)

		logits_shrunk = tf.layers.conv2d(hidden4_pool, filters = 1, kernel_size = 1, activation = None, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		logits = tf.image.resize_images(logits_shrunk, (input_height, input_width), method = 1)

		output_shrunk = tf.nn.sigmoid(logits_shrunk)
		output = tf.image.resize_images(output_shrunk, (input_height, input_width), method = 1)

		hidden5 = tf.layers.conv2d(hidden4_pool, filters = 128, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))		
		hidden5_pool = tf.keras.layers.AveragePooling2D()(hidden5)

		hidden6 = tf.layers.conv2d(hidden5_pool, filters = 128, kernel_size = 5, activation = tf.nn.leaky_relu, padding = "same", kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		hidden6_gap = tf.reduce_mean(hidden6, axis = [1, 2])

		gap_logits = tf.keras.layers.Dense(units = 1)(hidden6_gap)
		gap_output = tf.nn.sigmoid(gap_logits)

		return output, logits, gap_output, gap_logits

class Inpainter:
	def __init__(self, input_height, input_width):
		# placeholders
		self.input_height = input_height
		self.input_width = input_width

		self.real_images = tf.placeholder(tf.float32, shape = [None, input_height, input_width, 3])
		self.masks = tf.placeholder(tf.float32, shape = [None, input_height, input_width, 1])

		self.inpaint1 = increment_generator_first(self.real_images, self.masks, input_width, input_height, prefix = "gen_incr_1", reuse = None)
		self.confidence1, self.confidence_logits1, self.gap_confidence1, self.gap_confidence_logits1 = increment_discriminator(self.inpaint1, input_width, input_height, prefix = "dis_incr_1", reuse = None)

		self.inpaint2 = increment_generator_feedback(self.inpaint1, self.masks, self.confidence1, input_width, input_height, prefix = "gen_incr_2", reuse = None)
		self.confidence2, self.confidence_logits2, self.gap_confidence2, self.gap_confidence_logits2 = increment_discriminator(self.inpaint2, input_width, input_height, prefix = "dis_incr_2", reuse = None)

		self.real_confidence1, self.real_confidence_logits1, self.gap_real_confidence1, self.gap_real_confidence_logits1 = increment_discriminator(self.real_images, input_width, input_height, prefix = "dis_incr_1", reuse = True)
		self.real_confidence2, self.real_confidence_logits2, self.gap_real_confidence2, self.gap_real_confidence_logits2 = increment_discriminator(self.real_images, input_width, input_height, prefix = "dis_incr_2", reuse = True)

		# losses
		self.confidence1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits1, labels = util.soft_labels(1 - self.masks)), axis = [1, 2]) + tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits1, labels = util.soft_ones_like(self.gap_confidence1))
		self.inpaint1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits1, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits1, labels = util.soft_zeros_like(self.gap_confidence1)) + 800. * tf.reduce_mean(tf.pow(self.real_images - self.inpaint1, 2))

		self.confidence2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits2, labels = util.soft_labels(1 - self.masks)), axis = [1, 2]) + tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits2, labels = util.soft_ones_like(self.gap_confidence2))
		self.inpaint2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits2, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits2, labels = util.soft_zeros_like(self.gap_confidence2)) + 400. * tf.reduce_mean(tf.pow(self.real_images - self.inpaint2, 2))

		self.real1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.real_confidence_logits1, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_real_confidence_logits1, labels = util.soft_zeros_like(self.gap_real_confidence1))
		self.real2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.real_confidence_logits2, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_real_confidence_logits2, labels = util.soft_zeros_like(self.gap_real_confidence2))

		self.D1_loss = self.confidence1_loss + self.real1_loss
		self.D2_loss = self.confidence2_loss + self.real2_loss
		self.G1_loss = self.inpaint1_loss
		self.G2_loss = self.inpaint2_loss

		self.D1_loss_mean = tf.reduce_mean(self.D1_loss)
		self.D2_loss_mean = tf.reduce_mean(self.D2_loss)
		self.G1_loss_mean = tf.reduce_mean(self.G1_loss)
		self.G2_loss_mean = tf.reduce_mean(self.G2_loss)

		# trainers
		tvars = tf.trainable_variables()
		D1_vars = [var for var in tvars if "dis_incr_1" in var.name]
		D2_vars = [var for var in tvars if "dis_incr_2" in var.name]
		G1_vars = [var for var in tvars if "gen_incr_1" in var.name]
		G2_vars = [var for var in tvars if "gen_incr_2" in var.name]

		self.D1_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.D1_loss, var_list = D1_vars)
		self.D2_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.D2_loss, var_list = D2_vars)
		self.G1_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.G1_loss, var_list = G1_vars)
		self.G2_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.G2_loss, var_list = G2_vars)

	def generate_mask(self):
		return util.generate_box_mask(width = self.input_width, height = self.input_height, hole_size = 48, num_holes = 5)

class Inpainter_wo_feedback:
	def __init__(self, input_height, input_width):
		# placeholders
		self.input_height = input_height
		self.input_width = input_width

		self.real_images = tf.placeholder(tf.float32, shape = [None, input_height, input_width, 3])
		self.masks = tf.placeholder(tf.float32, shape = [None, input_height, input_width, 1])

		self.inpaint1 = increment_generator_first(self.real_images, self.masks, input_width, input_height, prefix = "gen_incr_1", reuse = None)
		self.confidence1, self.confidence_logits1, self.gap_confidence1, self.gap_confidence_logits1 = increment_discriminator(self.inpaint1, input_width, input_height, prefix = "dis_incr_1", reuse = None)

		self.inpaint2 = increment_generator_feedback(self.inpaint1, self.masks, self.masks, input_width, input_height, prefix = "gen_incr_2", reuse = None)
		self.confidence2, self.confidence_logits2, self.gap_confidence2, self.gap_confidence_logits2 = increment_discriminator(self.inpaint2, input_width, input_height, prefix = "dis_incr_2", reuse = None)

		self.real_confidence1, self.real_confidence_logits1, self.gap_real_confidence1, self.gap_real_confidence_logits1 = increment_discriminator(self.real_images, input_width, input_height, prefix = "dis_incr_1", reuse = True)
		self.real_confidence2, self.real_confidence_logits2, self.gap_real_confidence2, self.gap_real_confidence_logits2 = increment_discriminator(self.real_images, input_width, input_height, prefix = "dis_incr_2", reuse = True)

		# losses
		self.confidence1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits1, labels = util.soft_labels(1 - self.masks)), axis = [1, 2]) + tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits1, labels = util.soft_ones_like(self.gap_confidence1))
		self.inpaint1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits1, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits1, labels = util.soft_zeros_like(self.gap_confidence1)) + 800. * tf.reduce_mean(tf.pow(self.real_images - self.inpaint1, 2))

		self.confidence2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits2, labels = util.soft_labels(1 - self.masks)), axis = [1, 2]) + tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits2, labels = util.soft_ones_like(self.gap_confidence2))
		self.inpaint2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.confidence_logits2, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_confidence_logits2, labels = util.soft_zeros_like(self.gap_confidence2)) + 400. * tf.reduce_mean(tf.pow(self.real_images - self.inpaint2, 2))

		self.real1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.real_confidence_logits1, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_real_confidence_logits1, labels = util.soft_zeros_like(self.gap_real_confidence1))
		self.real2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.real_confidence_logits2, labels = util.soft_zeros_like(self.masks)), axis = [1, 2]) + 0.3 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.gap_real_confidence_logits2, labels = util.soft_zeros_like(self.gap_real_confidence2))

		self.D1_loss = self.confidence1_loss + self.real1_loss
		self.D2_loss = self.confidence2_loss + self.real2_loss
		self.G1_loss = self.inpaint1_loss
		self.G2_loss = self.inpaint2_loss

		self.D1_loss_mean = tf.reduce_mean(self.D1_loss)
		self.D2_loss_mean = tf.reduce_mean(self.D2_loss)
		self.G1_loss_mean = tf.reduce_mean(self.G1_loss)
		self.G2_loss_mean = tf.reduce_mean(self.G2_loss)

		# trainers
		tvars = tf.trainable_variables()
		D1_vars = [var for var in tvars if "dis_incr_1" in var.name]
		D2_vars = [var for var in tvars if "dis_incr_2" in var.name]
		G1_vars = [var for var in tvars if "gen_incr_1" in var.name]
		G2_vars = [var for var in tvars if "gen_incr_2" in var.name]

		self.D1_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.D1_loss, var_list = D1_vars)
		self.D2_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.D2_loss, var_list = D2_vars)
		self.G1_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.G1_loss, var_list = G1_vars)
		self.G2_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.G2_loss, var_list = G2_vars)

	def generate_mask(self):
		return util.generate_box_mask(width = self.input_width, height = self.input_height, hole_size = 48, num_holes = 5)