from numpy import array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import pickle
import warnings
import os
import json
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")

def load_doc(filename):
	with open(filename, 'r') as f_in:
		text = f_in.read()
	return text

def get_chars_mapping_input_lines(input_file):
	raw_text = load_doc(in_filename)
	lines = raw_text.split('\n')
	chars = sorted(list(set(raw_text)))
	mapping = dict((c, i) for i, c in enumerate(chars))
	return mapping, chars, lines

def get_sequences(mappper, lines):
	sequences = list()
	for line in lines:
		encoded_seq = [mappper[char] for char in line]
		sequences.append(encoded_seq)
	return sequences

def train_test_split(sequences, train_pct = 1):
	sequences = array(sequences)
	train_sequences, test_sequences = sequences[:int(len(sequences)*train_pct)], sequences[int(len(sequences)*train_pct):]
	sequences = train_sequences
	X, y = sequences[:,:-1], sequences[:,-1]
	return X, y, test_sequences

def one_hot_encoding(X, y, no_of_cat):
	X = array([to_categorical(x, num_classes=no_of_cat) for x in X])
	y = to_categorical(y, num_classes=no_of_cat)
	return X, y

def build_network(input_shape, output_size, params, model_name):
	model = Sequential()
	model.add(LSTM(75, input_shape=(input_shape[0], input_shape[1])))
	model.add(Dense(1024))
	model.add(Dense(output_size, activation='softmax'))
	model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params['metrics'])
	if not os.path.isdir(model_name):
		os.makedirs(model_name)
	with open(os.path.join(model_name, 'model_summary.txt'), 'w') as f:
		with redirect_stdout(f):
			model.summary()
	return model

def save_model(model, results, params, name):
	if not os.path.isdir(name):
		os.makedirs(name)
	model.save(os.path.join(name,'model.h5'))
	pickle.dump(mapping, open(os.path.join(name, 'mapping.pkl'), 'wb'))
	with open(os.path.join(name, 'training_history.json'), 'w') as json_out:
		json.dump(results.history, json_out, indent=2)
	with open(os.path.join(name, 'params.json'), 'w') as json_out:
		json.dump(params, json_out, indent=2)

# Parameters
hyperp = {}
in_filename = 'char_sequences.txt'
model_name = 'test_network_look_alike_070819_1900'
hyperp['loss'] = 'categorical_crossentropy'
hyperp['optimizer'] = 'adam'
hyperp['metrics'] = ['accuracy']
hyperp['training_percentage'] = 1
hyperp['epochs'] = 50
hyperp['batch_size'] = 2 ** 4

# Execution
mapping, chars, lines = get_chars_mapping_input_lines(in_filename)
vocab_size = len(mapping)
hyperp['vocabulary_size'] = vocab_size
seqs = get_sequences(mapping, lines)
hyperp['sequence_lenght'] = len(seqs[0]) - 1

X, y, _ = train_test_split(seqs, hyperp['training_percentage'])
X, y = one_hot_encoding(X, y, vocab_size)

char_rnn = build_network((X.shape[1], X.shape[2]), vocab_size, hyperp, model_name)

results = char_rnn.fit(X, y, epochs=hyperp['epochs'], verbose=1, batch_size=hyperp['batch_size'])

save_model(char_rnn, results, hyperp, model_name)
