
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy
import json
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def greedy_generator(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        yhat = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
    return in_text

def sampling_generator(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        yhat_probs = model.predict(encoded, verbose=0)
        yhat = numpy.random.choice(numpy.arange(0, len(mapping)), p=yhat_probs[0])
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
    return in_text


model_name = 'notes_network_160819_2130'
sample_len = 1000
sample_start = 'this is about'

model_path = os.path.join('data', model_name)
model = load_model(os.path.join(model_path, 'model.h5'))
mapping = load(open(os.path.join(model_path,'mapping.pkl'), 'rb'))
with open(os.path.join(model_path, 'params.json'), 'r') as p_in:
    params = json.load(p_in)

greedy = True
sampling = True

if greedy:
    g_sample = greedy_generator(model, mapping, params['sequence_lenght'], sample_start, sample_len)
    print('\n', 15 * '-', ' Greedy sample starting here')
    print(g_sample)
    g_sample_name = 'sample_' + str(datetime.datetime.now()).replace(' ','T') + '.txt'
    with open(os.path.join(model_path, g_sample_name), 'w') as s_out:
        s_out.write('Greedy sample: ' + g_sample)

if sampling:
    s_sample = sampling_generator(model, mapping, params['sequence_lenght'], sample_start, sample_len)
    print('\n', 15 * '-', ' Sampling sample starting here')
    print(s_sample)
    s_sample_name = 'sample_' + str(datetime.datetime.now()).replace(' ','T') + '.txt'
    with open(os.path.join(model_path, s_sample_name), 'w') as s_out:
        s_out.write('Sampling sample: ' + s_sample)





