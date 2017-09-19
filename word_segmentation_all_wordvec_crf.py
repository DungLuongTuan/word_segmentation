"""
	-	Bi-LSTM model for vietnamese word segmentation
	-	1 layer B-LSTM -> 1 layer CRF
	-	parameters:
		+	epochs = 1000
		+ 	batch size = 10
		+	n_hidden (demension of each hidden state also cell state) = 100
		+	max length of input sequences = 500
	-	models save into ./models/crf/modelx.ckpt where x is epoch
	-	loss save into ./models/crf/loss
"""

import numpy as np 
import tensorflow as tf 
import fasttext as ft 
from os import listdir
from load_data import load_data
import pickle
import sys

tag = ['BB', 'MM', 'EE']
Word2vec = ft.load_model('./pretrain_model/word2vec/vi.bin')

def main():
	### model parameters
	n_hidden = 100
	max_state = 500
	epochs = 1000
	batch_size = int(sys.argv[1])
	### graph
	x = tf.placeholder(tf.float32, [None, max_state, n_hidden])
	y = tf.placeholder(tf.int32, [None, max_state])
	sequence_length = tf.placeholder(tf.int32, [None])
	w = tf.Variable(tf.truncated_normal([2*n_hidden, 4]), name = 'w')
	b = tf.Variable(tf.truncated_normal([1, 4]), name = 'b')
	fw_cell_lstm = tf.contrib.rnn.LSTMCell(n_hidden)
	bw_cell_lstm = tf.contrib.rnn.LSTMCell(n_hidden)
	(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_lstm, cell_bw = bw_cell_lstm, sequence_length = sequence_length, inputs = x, dtype = tf.float32)
	h = tf.concat([output_fw, output_bw], axis = -1)
	h_slice = tf.reshape(h, [-1, 2*n_hidden])
	output_slice = tf.matmul(h_slice, w) + b
	output = tf.reshape(output_slice, [-1, max_state, 4])

	log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(output, y, sequence_length)

	loss = tf.reduce_mean(-log_likelihood)
	optimizer = tf.train.RMSPropOptimizer(float(sys.argv[2])).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	saver = tf.train.Saver(max_to_keep = 1000)
	f_loss = open('./models/crf/loss', 'w')
	for epoch in range(epochs):
		f = open('./Vietnamese Word Segmentation/1_file_data/train_corpus', 'r')
		f_matrix = open('./models/crf/' + 'transition_matrix' + str(int(epoch)), 'wb')
		start = 0
		cnt = 0
		sum_loss = 0
		while (start < 53204):
			print('batch: ', cnt, end = '\r')
			cnt += 1
			start += batch_size
			batch_data, batch_labels, batch_seq = load_data(batch_size, f, Word2vec)
			pred_output, transition_matrix, loss_, _ = sess.run([output, transition_params, loss, optimizer], feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
			sum_loss += loss_
			pickle.dump(transition_matrix, f_matrix)
		print('epoch: ', epoch, ' loss: ', sum_loss)
		f_loss.write('epoch: ' + str(int(epoch)) + ' - loss: ' + str(sum_loss) + '\n')
		saver.save(sess, './models/crf/' + 'model' + str(int(epoch)) + '.ckpt')
		f.close()
		f_matrix.close()
	f_loss.close()

if __name__ == '__main__':
	main()