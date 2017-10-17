import numpy as np 
import tensorflow as tf 
import fasttext as ft 
from os import listdir
import sys

tag = ['BB', 'MM', 'EE']
Word2vec = ft.load_model('vi.bin')

def load_data(batch_size, f):
	max_state = 500
	seg_data = []
	seg_labels = []
	seg_sequence_length = []
	sen_in_batch = 1
	for row in f:
		row_data = []
		row_labels = []
		row_list = row[:-1].split(' ')
		row_length = 0
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
				for i in range(1, len(split_word) - 1):
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
		sen_in_batch += 1
		if (sen_in_batch > batch_size):
			break
	return seg_data, seg_labels, seg_sequence_length

def main():
	### model parameters
	n_hidden = 100 # dimention of each hidden state also cell state
	max_state = 500 # max length of sequences
	### build graph
	print('build graph')
	x = tf.placeholder(tf.float32, [None, max_state, 100])
	y = tf.placeholder(tf.float32, [None, max_state, 3])
	sequence_length = tf.placeholder(tf.int32, [None])
	w = tf.get_variable('w', shape = [2*n_hidden, 3])
	b = tf.get_variable('b', shape = [1, 3])
	fw_cell_lstm = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	bw_cell_lstm = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_lstm, cell_bw = bw_cell_lstm, sequence_length = sequence_length, inputs = x, dtype = tf.float32)
	h = tf.concat([output_fw, output_bw], axis = -1)
	h_relu = tf.nn.relu(h)
	h_slice = tf.reshape(h_relu, [-1, 2*n_hidden])
	output_slice = tf.nn.softmax(tf.matmul(h_slice, w) + b)
	output = tf.reshape(output_slice, [-1, max_state, 3])

	### evaluation on test data
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, sys.argv[1] + '/model' + sys.argv[2] + '.ckpt')
	print('evaluation on train data')
	f = open('./Vietnamese Word Segmentation/random_corpus/' + sys.argv[3] + '/train_corpus', 'r')
	sum_cnt = -1
	cnt = -1
	num_sen = 0
	while (num_sen < 61377):
		num_sen += 1
		print('sentence: ', num_sen, '/', 61377, end = '\r')
		seg_data, seg_labels, seg_sequence_length = load_data(1, f)
		result = sess.run(output, feed_dict = {x: seg_data, sequence_length: seg_sequence_length})
		true_tag = np.zeros(3)
		pred_tag = np.zeros(3)
		for j in range(seg_sequence_length[0]):
			if (seg_labels[0][j].tolist() == [1, 0, 0]):
				sum_cnt += 1
				if (true_tag.tolist() == pred_tag.tolist()):
					cnt += 1
				true_tag = np.zeros(3)
				pred_tag = np.zeros(3)
			true_tag += seg_labels[0][j]
			mini_pred = np.zeros(3)
			mini_pred[np.argmax(result[0][j])] = 1
			pred_tag += mini_pred
	print(cnt, ' / ', sum_cnt)
	print(cnt/sum_cnt)
	f.close()

if __name__ == '__main__':
	main()