import numpy
from numpy import array
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Activation, TimeDistributed
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

def train_test_split(sequences, train_pct = 1, shuffle: bool = True):
	sequences = array(sequences)
	if shuffle:
		numpy.random.shuffle(sequences)
	train_sequences, test_sequences = sequences[:int(len(sequences)*train_pct)], sequences[int(len(sequences)*train_pct):]
	X_train, y_train = train_sequences[:,:-1], train_sequences[:,-1]
	X_test, y_test = test_sequences[:,:-1], test_sequences[:,-1]
	return X_train, y_train, X_test, y_test

def one_hot_encoding(X, y, no_of_cat):
	X = array([to_categorical(x, num_classes=no_of_cat) for x in X])
	y = to_categorical(y, num_classes=no_of_cat)
	return X, y

def build_network(params, model_name):
	model = Sequential()
	model.add(GRU(512, return_sequences=True, input_shape=(params['sequence_lenght'], params['vocabulary_size'])))
	model.add(Dropout(0.2))
	model.add(GRU(512, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(params['vocabulary_size']))
	model.add(Activation('softmax'))
	model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params['metrics'])
	if not os.path.isdir(os.path.join('data', model_name)):
		os.makedirs(os.path.join('data', model_name))
	with open(os.path.join('data', model_name, 'model_summary.txt'), 'w') as f:
		with redirect_stdout(f):
			print(model.summary())
	return model

def model_probability(X, y, model):
	predictions = model.predict(X)
	true_mask = y.astype(bool)
	return predictions[true_mask]

def perplexity(distribution):
	entropy = - numpy.sum(numpy.log(distribution) / len(distribution))
	return numpy.power(2, entropy)

def save_model(model, results, params, name):
	if not os.path.isdir(os.path.join('data', name)):
		os.makedirs(os.path.join('data', name))
	model.save(os.path.join('data', name, 'model.h5'))
	pickle.dump(mapping, open(os.path.join('data', name, 'mapping.pkl'), 'wb'))
	with open(os.path.join('data', name, 'training_history.json'), 'w') as json_out:
		json.dump(results.history, json_out, indent=2)
	with open(os.path.join('data', name, 'params.json'), 'w') as json_out:
		json.dump(params, json_out, indent=2)


# Parameters
hyperp = {}
in_filename = os.path.join('data', 'char_sequences.txt')
model_name = 'notes_network_100819_2230'
hyperp['loss'] = 'categorical_crossentropy'
hyperp['optimizer'] = 'adam'
hyperp['metrics'] = ['accuracy']
hyperp['training_percentage'] = 0.95
assert 0 < hyperp['training_percentage'] < 1, 'Training percentage must be value between 0 and 1'
hyperp['epochs'] = 5
hyperp['batch_size'] = 2 ** 16
hyperp['random_seed'] = 1

# Execution
numpy.random.seed(hyperp['random_seed'])
mapping, chars, lines = get_chars_mapping_input_lines(in_filename)
vocab_size = len(mapping)
hyperp['vocabulary_size'] = vocab_size
seqs = get_sequences(mapping, lines)
hyperp['sequence_lenght'] = len(seqs[0]) - 1

X_train, y_train, X_test, y_test = train_test_split(seqs, hyperp['training_percentage'])
X_train, y_train = one_hot_encoding(X_train, y_train, vocab_size)
char_rnn = build_network(hyperp, model_name)
results = char_rnn.fit(X_train, y_train, epochs=hyperp['epochs'], verbose=1, batch_size=hyperp['batch_size'])

X_test, y_test = one_hot_encoding(X_test, y_test, vocab_size)
hyperp['perplexity'] = {
	'training': perplexity(model_probability(X_train, y_train, char_rnn)),
	'test': perplexity(model_probability(X_test, y_test, char_rnn))}

save_model(char_rnn, results, hyperp, model_name)