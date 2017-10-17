import numpy as np 
import tensorflow as tf 
import fasttext as ft 
from os import listdir
import pickle
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
				row_labels.append(0)
			if (len(split_word) > 1):
				row_data.append(Word2vec[split_word[0]])
				row_labels.append(0)
				for i in range(1, len(split_word) - 1):
					row_data.append(Word2vec[split_word[i]])
					row_labels.append(1)
				row_data.append(Word2vec[split_word[-1]])
				row_labels.append(2)
		while (len(row_data) < max_state):
			row_data.append(np.zeros(100))
			row_labels.append(3)
		seg_data.append(row_data)
		seg_labels.append(row_labels)
		seg_sequence_length.append(row_length)
		sen_in_batch += 1
		if (sen_in_batch > batch_size):
			break
	return seg_data, seg_labels, seg_sequence_length

def main():
	### model parameters
	n_hidden = 100
	max_state = 500
	x = tf.placeholder(tf.float32, [None, max_state, n_hidden])
	y = tf.placeholder(tf.int32, [None, max_state])
	sequence_length = tf.placeholder(tf.int32, [None])
	w = tf.get_variable(shape = [2*n_hidden, 4], name = 'w')
	b = tf.get_variable(shape = [1, 4], name = 'b')
	fw_cell_lstm = tf.contrib.rnn.LSTMCell(n_hidden)
	bw_cell_lstm = tf.contrib.rnn.LSTMCell(n_hidden)
	(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_lstm, cell_bw = bw_cell_lstm, sequence_length = sequence_length, inputs = x, dtype = tf.float32)
	h = tf.concat([output_fw, output_bw], axis = -1)
	h_slice = tf.reshape(h, [-1, 2*n_hidden])
	output_slice = tf.matmul(h_slice, w) + b
	output = tf.reshape(output_slice, [-1, max_state, 4])

	sess = tf.InteractiveSession()
	saver = tf.train.Saver(max_to_keep = 1000)
	saver.restore(sess, sys.argv[1] + '/model' + sys.argv[2] + '.ckpt')
	### evaluate on training data
	print('evaluate on testing data')
	f = open('./Vietnamese Word Segmentation/random_corpus/' + sys.argv[3] + '/test_corpus', 'r')
	f_matrix = open(sys.argv[1] + '/transition_matrix' + sys.argv[2], 'rb')
	transition_matrix = pickle.load(f_matrix)
	start = 0
	cnt = 0
	sum_cnt = 0
	while (start < 6820):
		print('batch: ', start, end = '\r')
		batch_data, batch_labels, batch_seq = load_data(1, f)
		start += 1
		predict_output = sess.run(output, feed_dict = {x: batch_data, sequence_length: batch_seq})
		predict_output = predict_output[0][:batch_seq[0]]
		predict_labels, _ = tf.contrib.crf.viterbi_decode(predict_output, transition_matrix)
		batch_labels = batch_labels[0][:batch_seq[0]]
		predict_labels.append(0)
		batch_labels.append(0)
		dd = 0
		for i in range(1, batch_seq[0] + 1):
			if (batch_labels[i] == 0):
				sum_cnt += 1
				p_labels = predict_labels[dd:i]
				t_labels = batch_labels[dd:i]
				if (p_labels == t_labels):
					cnt += 1
				dd = i
	print(cnt, ' / ', sum_cnt)
	print(cnt/sum_cnt)
	f_matrix.close()
	f.close()

if __name__ == '__main__':
	main()
