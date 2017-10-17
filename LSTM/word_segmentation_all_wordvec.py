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
	epochs = 50
	batch_size = int(sys.argv[1])
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
	optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

	### run model
	sess = tf.InteractiveSession()
	saver = tf.train.Saver(max_to_keep = 50)
	tf.global_variables_initializer().run()
	f_loss = open(sys.argv[2] + '/loss', 'w')
	for epoch in range(epochs):
		f = open('./Vietnamese Word Segmentation/random_corpus/' + sys.argv[3] + '/train_corpus', 'r')
		start = 0
		sum_loss = 0
		cnt = 0
		while (start < 61377):
			print('batch: ', cnt, end = '\r')
			batch_data, batch_labels, batch_seq = load_data(batch_size, f)
			start += batch_size
			cnt += 1
			sess.run(optimizer, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
			sum_loss += sess.run(loss, feed_dict = {x: batch_data, y: batch_labels, sequence_length: batch_seq})
		print('epoch: ', epoch, ' loss: ', sum_loss)
		f_loss.write('epoch: ' + str(epoch) + ' loss: ' + str(sum_loss) + '\n')
		saver.save(sess, sys.argv[2] + '/model' + str(int(epoch)) + '.ckpt')
		f.close()
	f_loss.close()

if __name__ == '__main__':
	main()