"""
	-	Bi-LSTM model for vietnamese word segmentation
	-	1 layer B-LSTM -> 1 layer relu -> 1 layer softmax
	-	parameters:
		+	epochs = 10
		+ 	batch size = 10
		+	n_hidden (demension of each hidden state also cell state) = 100
		+	max length of input sequences = 500
	-	models save into ./models/word2vec/modelx.ckpt where x is epoch
	-	loss save into ./models/word2vec/loss
"""

import numpy as np 
import tensorflow as tf 
import fasttext as ft 
from os import listdir
import sys

tag = ['BB', 'MM', 'EE']
Word2vec = ft.load_model('vi.bin')

def main():
	### model parameters
	epochs = 100
	batch_size = 10
	n_hidden = 100 # dimension of each hidden state also cell state
	max_state = 500 # max length of sequences
	### build graph
	print('build graph')
	x = tf.placeholder(tf.float32, [None, max_state, 100])
	y = tf.placeholder(tf.float32, [None, max_state, 3])
	sequence_length = tf.placeholder(tf.int32, [None])
	w = tf.Variable(tf.truncated_normal([2*n_hidden, 3]), name = 'w')
	b = tf.Variable(tf.truncated_normal([1, 3]), name = 'b')
	fw_cell_lstm = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	bw_cell_lstm = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_lstm, cell_bw = bw_cell_lstm, sequence_length = sequence_length, inputs = x, dtype = tf.float32)
	h = tf.concat([output_fw, output_bw], axis = -1)
	h_relu = tf.nn.relu(h)
	h_slice = tf.reshape(h_relu, [-1, 2*n_hidden])
	output_slice = tf.nn.softmax(tf.matmul(h_slice, w) + b)
	output = tf.reshape(output_slice, [-1, max_state, 3])

	loss = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(y*tf.log(output), reduction_indices = [2]), reduction_indices = [1]))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

	### run model
	sess = tf.InteractiveSession()
	saver = tf.train.Saver(max_to_keep = 50)
	tf.global_variables_initializer().run()
	f_loss = open('./models/word2vec/loss', 'w')

	for epoch in range(epochs):
		f = open('./Vietnamese Word Segmentation/1_file_data/train_corpus', 'r')
		start = 0
		sum_loss = 0
		cnt = 0
		while (start <= 53203):
			print('batch: ', cnt, end = '\r')
			batch_data, batch_labels, batch_seq = load_data(batch_size, f, Word2vec)
			start += batch_size
			cnt += 1
			sess.run(optimizer, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
			sum_loss += sess.run(loss, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
		print('epoch: ', epoch, ' loss: ', sum_loss)
		f_loss.write('epoch: ' + str(int(epoch)) + ' - loss: ' + str(sum_loss) + '\n')
		saver.save(sess, './models/word2vec/' + 'model' + str(int(epoch)) + '.ckpt')
		f.close()
	f_loss.close()

if __name__ == '__main__':
	main()