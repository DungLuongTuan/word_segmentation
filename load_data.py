import numpy as np 

def load_data(batch_size, f, Word2vec):
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