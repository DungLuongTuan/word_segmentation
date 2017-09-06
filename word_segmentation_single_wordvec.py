"""
	- Use word2vec model for embedding input
	- read files data during training model
"""

import numpy as np 
import tensorflow as tf 
import fasttext as ft 
from os import listdir

tag = ['BB', 'MM', 'EE']
Word2vec = ft.load_model('vi.bin')

def load_data(path):
	max_state = 500
	seg_data = []
	seg_labels = []
	seg_sequence_length = []
	f = open(path, 'r', encoding="utf8")
	for row in f:
		row_data = []
		row_labels = []
		row_list = row[:-1].split(' ')
		row_length = 0
		if (row[0] == '<'):
			continue
		for word in row_list:
			split_word = word.split('_')
			row_length += len(split_word)
			if (len(split_word) == 1):
				row_data.append(Word2vec[split_word[0]])
				label = np.array([1, 0, 0])
				row_labels.append(label)
			if (len(split_word) > 1):
				row_data.append(Word2vec[split_word[0]])
				label = np.array([1, 0, 0])
				row_labels.append(label)
				for i in range(1, len(split_word) - 2):
					row_data.append(Word2vec[split_word[i]])
					label = np.array([0, 1, 0])
					row_labels.append(label)
				row_data.append(Word2vec[split_word[-1]])
				label = np.array([0, 0, 1])
				row_labels.append(label)
		while (len(row_data) < max_state):
			row_data.append(np.zeros(100))
			row_labels.append(np.zeros(3))
		seg_data.append(row_data)
		seg_labels.append(row_labels)
		seg_sequence_length.append(row_length)
	f.close()
	return seg_data, seg_labels, seg_sequence_length

def main():
	### model's parameters
	epochs = 20
	batch_size = 320
	n_hidden = 100 # dimention of each hidden state also cell state
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
	# tf.global_variables_initializer().run()
	train_file_names = listdir('./Vietnamese Word Segmentation/train_corpus')
	for epoch in range(epochs):
		cnt = 0
		sum_loss = 0
		for file_name in train_file_names:
			seg_data, seg_labels, seg_sequence_length = load_data('./Vietnamese Word Segmentation/train_corpus/' + file_name)
			start = 0
			cnt += 1
			print('file: ', cnt, '/', len(train_file_names), end = '\r')
			while (start <= len(seg_sequence_length) - 1):
				batch_data = seg_data[start : min(start + batch_size, len(seg_sequence_length))]
				batch_labels = seg_labels[start : min(start + batch_size, len(seg_sequence_length))]
				batch_seq = seg_sequence_length[start : min(start + batch_size, len(seg_sequence_length))]
				start += batch_size
				sess.run(optimizer, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
				sum_loss += sess.run(loss, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
		print('epoch: ', epoch, ' loss: ', sum_loss)
		saver.save(sess, 'model' + str(epoch) + '.ckpt')

	### evaluate on training data
	print('evaluate on training data')
	train_file_names = listdir('./Vietnamese Word Segmentation/train_corpus')
	sum_cnt = -1
	cnt = -1
	file = 0
	for file_name in train_file_names:
		file += 1
		print('file: ', file, '/', len(train_file_names), end = '\r')
		seg_data, seg_labels, seg_sequence_length = load_data('./Vietnamese Word Segmentation/train_corpus/' + file_name)
		for i in range(len(seg_sequence_length)):
			result = sess.run(output, feed_dict = {x: np.array([seg_data[i]]), sequence_length: [seg_sequence_length[i]]})
			true_tag = np.zeros(3)
			pred_tag = np.zeros(3)
			for j in range(seg_sequence_length[i]):
				if (seg_labels[i][j].tolist() == [1, 0, 0]):
					sum_cnt += 1
					if (true_tag.tolist() == pred_tag.tolist()):
						cnt += 1
					true_tag = np.zeros(3)
					pred_tag = np.zeros(3)
				true_tag += seg_labels[i][j]
				mini_pred = np.zeros(3)
				mini_pred[np.argmax(result[0][j])] = 1
				pred_tag += mini_pred
	print(cnt, ' / ', sum_cnt)
	print(cnt/sum_cnt)
	### evaluation on test data
	print('evaluation on test data')
	test_file_names = listdir('./Vietnamese Word Segmentation/test_corpus')
	sum_cnt = -1
	cnt = -1
	file = 0
	for file_name in test_file_names:
		file += 1
		print('file: ', file, '/', len(test_file_names), end = '\r')
		seg_test_data, seg_test_labels, seg_test_sequence_length = load_data('./Vietnamese Word Segmentation/test_corpus/' + file_name)
		for i in range(len(seg_test_sequence_length)):
			result = sess.run(output, feed_dict = {x: np.array([seg_test_data[i]]), sequence_length: [seg_test_sequence_length[i]]})
			true_tag = np.zeros(3)
			pred_tag = np.zeros(3)
			for j in range(seg_test_sequence_length[i]):
				if (seg_test_labels[i][j].tolist() == [1, 0, 0]):
					sum_cnt += 1
					if (true_tag.tolist() == pred_tag.tolist()):
						cnt += 1
					true_tag = np.zeros(3)
					pred_tag = np.zeros(3)
				true_tag += seg_test_labels[i][j]
				mini_pred = np.zeros(3)
				mini_pred[np.argmax(result[0][j])] = 1
				pred_tag += mini_pred
	print(cnt, ' / ', sum_cnt)
	print(cnt/sum_cnt)

if __name__ == '__main__':
	main()