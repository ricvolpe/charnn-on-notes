
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
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


model_name = 'test_network_look_alike_190819_1700'
sample_len = 500
sample_start = 'this is about'

model_path = os.path.join('data', model_name)
model = load_model(os.path.join(model_path, 'model.h5'))
mapping = load(open(os.path.join(model_path,'mapping.pkl'), 'rb'))
with open(os.path.join(model_path, 'params.json'), 'r') as p_in:
    params = json.load(p_in)

sample = generate_seq(model, mapping, params['sequence_lenght'], sample_start, sample_len)
print('\n', 15 * '-', 'Sample starting here')
print(sample)

with open(os.path.join(model_path, 'sample.txt'), 'w') as s_out:
    s_out.write(sample)