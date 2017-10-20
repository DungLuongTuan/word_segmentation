import numpy as np 
import tensorflow as tf 
import math
import pickle

class CNN_BLSTM_CRF(object):
	"""
		convolutional neural network for capturing character level information
		Bi-directional Long Short Term Memory for capturing relation between words in a sentences
		Conditional Random Field for capturing relation between labels in sequence labels corresponding to sentence
	"""
	
	def __init__(self, max_length_word, max_length_sentence, char_embedding_size, window_size, number_conv_units, n_hidden, num_tag, char_dictionary, percent_GPU, work):
		"""
			model's parameters
			max_length_word = 15
			max_length_sentence = 500
			char_embedding_size = 25
			window_size = 3
			number_conv_units = 50
			n_hidden = 100
			num_tag = 4
			batch_size = 10
		"""
		self.max_length_word = max_length_word
		self.max_length_sentence = max_length_sentence
		self.char_embedding_size = char_embedding_size
		self.window_size = window_size
		self.number_conv_units = number_conv_units
		self.n_hidden = n_hidden
		self.num_tag = num_tag
		self.char_dictionary = char_dictionary
		self.work = work
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = percent_GPU
		self.sess = tf.Session(config = config)
		self.build_graph()


	def build_graph(self):
		### combine all parts of the main graph
		if (self.work == 'train'):
			self.init_placeholder()
			self.init_variables()
			self.get_character_level_information()
			self.build_BLSTM_CRF_model()
			self.loss_optimizer()
			self.sess.run(tf.global_variables_initializer())
		else:
			self.init_placeholder()
			self.init_variables_eval()
			self.get_character_level_information()
			self.build_BLSTM_CRF_model()


	def init_placeholder(self):
		### character presentation placeholder: [batch_size, max_length_sentence, max_length_word]
		self.char_input = tf.placeholder(tf.int32, [None, self.max_length_sentence, self.max_length_word])
		### word presentation placeholder: [batch_size, max_length_sentence, n_hidden]
		self.word_input = tf.placeholder(tf.float32, [None, self.max_length_sentence, self.n_hidden])
		### output tags placeholder: [batch_size, max_length_sentence]
		self.y = tf.placeholder(tf.int32, [None, self.max_length_sentence])
		### list length of each sentence in batch
		self.sequence_length = tf.placeholder(tf.int32, [None])
		### learning rate
		self.lr = tf.placeholder(tf.float32, None)
		### get batch size
		self.batch_size = tf.shape(self.char_input)[0]


	def init_variables(self):
		### character embedding: [len(char_dictionary), char_embedding_size]
		self.initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype = tf.float32)
		self.char_embedding = tf.get_variable(name = 'character_embedding_matrix', shape = [len(self.char_dictionary), self.char_embedding_size], initializer = self.initializer, dtype = tf.float32)
		### convolutional layer variables
		self.w_conv = tf.get_variable(name = 'w_conv', shape = [self.char_embedding_size*self.window_size, self.number_conv_units], dtype = tf.float32)
		self.b_conv = tf.get_variable(name = 'b_conv', shape = [self.number_conv_units], dtype = tf.float32)
		### BLSTM + CRF variables
		self.w = tf.Variable(tf.truncated_normal([2*self.n_hidden, self.num_tag]), name = 'w_lstm')
		self.b = tf.Variable(tf.truncated_normal([1, self.num_tag]), name = 'b_lstm')


	def init_variables_eval(self):
		### character embedding: [len(char_dictionary), char_embedding_size]
		self.char_embedding = tf.get_variable(name = 'character_embedding_matrix', shape = [len(self.char_dictionary), self.char_embedding_size], dtype = tf.float32)
		### convolutional layer variables
		self.w_conv = tf.get_variable(name = 'w_conv', shape = [self.char_embedding_size*self.window_size, self.number_conv_units], dtype = tf.float32)
		self.b_conv = tf.get_variable(name = 'b_conv', shape = [self.number_conv_units], dtype = tf.float32)
		### BLSTM + CRF variables
		self.w = tf.get_variable(name = 'w_lstm', shape = [2*self.n_hidden, self.num_tag], dtype = tf.float32)
		self.b = tf.get_variable(name = 'b_lstm', shape = [1, self.num_tag], dtype = tf.float32)


	def get_character_level_information(self):
		### transform input to character embedding matrix (CNN layer)
		### char_input_embedded: [batch_size, max_sequence_length, max_length_word, char_embedding_size]
		self.char_input_slice = tf.reshape(self.char_input, [self.batch_size*self.max_length_sentence*self.max_length_word])
		self.char_input_embedded = tf.nn.embedding_lookup(params = self.char_embedding, ids = self.char_input_slice)
		self.char_input_embedded = tf.reshape(self.char_input_embedded, [self.batch_size, self.max_length_sentence, self.max_length_word, self.char_embedding_size])
		### feed character input embedding to convolutional layer
		# preprocess to compute Zm (concatenation of 'windown_size' consecutive character embeddings)
		self.char_input_embedded_S = tf.reshape(self.char_input_embedded, [-1, self.max_length_word, self.char_embedding_size])
		self.char_input_embedded_S_T = tf.transpose(self.char_input_embedded_S, [0, 2, 1])
		self.char_input_embedded_S_T_S = tf.reshape(self.char_input_embedded_S_T, [-1, self.max_length_word])
		self.char_input_embedded_S_T_S_T = tf.transpose(self.char_input_embedded_S_T_S, [1, 0])
		# get Zm
		self.val = self.char_input_embedded_S_T_S_T[:-(self.window_size - 1)]
		self.i = tf.constant(1)
		def condition(val, i):
			return tf.less(i, self.window_size)
		def body(val, i):
			val = tf.concat([val, self.char_input_embedded_S_T_S_T[i:self.max_length_word - (self.window_size - 1 - i)]], axis = -1)
			i = tf.add(i, 1)
			return [val, i]
		self.Zm_S_T_S_T, _ = tf.while_loop(cond = condition, body = body, loop_vars = [self.val, self.i], back_prop = True, shape_invariants = [tf.TensorShape([self.max_length_word - self.window_size + 1, None]), self.i.get_shape()])
		# change Zm back to the correct form [batch_size, max_length_sentence, max_length_word - window_size + 1, window_size*char_embedding_size]
		self.Zm_S_T_S = tf.transpose(self.Zm_S_T_S_T, [1, 0])
		self.Zm_S_T = tf.reshape(self.Zm_S_T_S, [-1, self.char_embedding_size*self.window_size, self.max_length_word - self.window_size + 1])
		self.Zm_S = tf.transpose(self.Zm_S_T, [0, 2, 1])
		self.Zm = tf.reshape(self.Zm_S, [self.batch_size, self.max_length_sentence, self.max_length_word - self.window_size + 1, self.char_embedding_size*self.window_size])
		# get character level embedding
		# Zm_slice: [batch_size*max_length_sentence*(max_length_word - window_size + 1), char_embed_size*window_size]
		self.Zm_slice = tf.reshape(self.Zm, [-1, self.char_embedding_size*self.window_size])
		self.conv_matrix_slice = tf.matmul(self.Zm_slice, self.w_conv) + self.b_conv
		# conv_matrix: [batch_size, max_length_sentence, max_length_word - window_size + 1, number_conv_units]
		self.conv_matrix = tf.reshape(self.conv_matrix_slice, [self.batch_size, self.max_length_sentence, self.max_length_word - self.window_size + 1, self.number_conv_units])
		# conv_vector: [batch_size, max_sentence_length, number_conv_units]
		self.conv_vector = tf.reduce_max(self.conv_matrix, axis = -2)


	def build_BLSTM_CRF_model(self):
		### BLSTM + CRF layer
		### input of LSTM is concatenation of word embedding + character embedding
		self.x = tf.concat([self.word_input, self.conv_vector], axis = -1)
		### define forward and backward lstm cell
		self.cell_fw = tf.contrib.rnn.LSTMCell(num_units = self.n_hidden)
		self.cell_bw = tf.contrib.rnn.LSTMCell(num_units = self.n_hidden)
		### BLSTM layer
		(self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.cell_fw, cell_bw = self.cell_bw, inputs = self.x, sequence_length = self.sequence_length, dtype = tf.float32)
		### concatenation of h_fw and h_bw: [batch_size, max_length_sentence, 2*n_hidden]
		self.h = tf.concat([self.output_fw, self.output_bw], axis = -1)
		### slice h: [batch_size*max_length_sentence, 2*n_hidden]
		self.h_slice = tf.reshape(self.h, [-1, 2*self.n_hidden])
		### reshape h to [batch_size, max_length_sentence, num_tags]
		self.output_slice = tf.matmul(self.h_slice, self.w) + self.b
		self.output = tf.reshape(self.output_slice, [-1, self.max_length_sentence, self.num_tag])
		### CRF layer
		self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.output, self.y, self.sequence_length)
	

	def loss_optimizer(self):
		self.loss = tf.reduce_mean(-self.log_likelihood)
		self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


	def train_new_model(self, char_input, word_input, labels, sequence_length, lr):
		loss, self.transition_matrix, _ = self.sess.run([self.loss, self.transition_params, self.optimizer], feed_dict = {self.char_input: char_input, self.word_input: word_input, self.y: labels, self.sequence_length: sequence_length, self.lr: lr})
		return loss


	def save_model(self, save_path, epoch):
		saver = tf.train.Saver(max_to_keep = 1000)
		### save variables of graph
		saver.save(self.sess, save_path + '/model' + str(epoch) + '.ckpt')
		### save transition matrix
		f = open(save_path + '/transition_matrix' + str(epoch), 'wb')
		pickle.dump(self.transition_matrix, f)
		f.close()


	def load_model(self, load_path, epoch):
		saver = tf.train.Saver(max_to_keep = 1000)
		saver.restore(self.sess, load_path + '/model' + str(epoch) + '.ckpt')
		f = open(load_path + '/transition_matrix' + str(epoch), 'rb')
		self.transition_matrix = pickle.load(f)
		f.close()


	def predict(self, char_input, word_input, sequence_length):
		predict_output = self.sess.run(self.output, feed_dict = {self.char_input: char_input, self.word_input: word_input, self.sequence_length: sequence_length})
		predict_output = predict_output[0][:sequence_length[0]]
		predict_labels, _ = tf.contrib.crf.viterbi_decode(predict_output, self.transition_matrix)
		return predict_labels


def main():
	model = CNN_BLSTM_CRF(max_length_word = 15, max_length_sentence = 500, char_embedding_size = 25, window_size = 3, number_conv_units = 50)
	# model.make_char_dictionary('./Vietnamese Word Segmentation/1_file_data/train_corpus', './model/cnn_lstm_crf/small_data/char_dictionary')
	model.load_dictionary('model/cnn_lstm_crf/small_data/char_dictionary')
	model.train_new_model('Vietnamese Word Segmentation/1_file_data/small_train_corpus')

if __name__ == '__main__':
	main()


