import os
import string

def load_doc(filename):
	with open(filename, 'r') as f_in:
		text = f_in.read()
	return text

DATA_CHOICES = {
	'naive_test': 'data/rhyme.txt',
	'look_alike_test': '/Users/Ric/_personal/_personal_data/mac-notes/personal_notes_test_data.txt',
	'personal_notes': '/Users/Ric/_personal/_personal_data/mac-notes/all_my_personal_notes.txt'
}

raw_text = load_doc(DATA_CHOICES['personal_notes'])
tokens = raw_text.split()
raw_text = ' '.join(tokens).lower()

vocab_str = ' ' + string.ascii_letters + ''.join([str(x) for x in range(10)]) + string.punctuation
raw_text = ''.join([x for x in raw_text if x in vocab_str])

length = 100
sequences = list()
for i in range(length, len(raw_text)):
	seq = raw_text[i-length:i+1]
	sequences.append(seq)

def save_doc(lines, filename):
	data = '\n'.join(lines)
	with open(filename, 'w') as f_out:
		f_out.write(data)

out_filename = os.path.join('data', 'char_sequences.txt')
save_doc(sequences, out_filename)