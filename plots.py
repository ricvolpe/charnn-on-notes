import matplotlib.pyplot as plt
import os
import json

model_name = 'notes_network_160819_2130'

model_path = os.path.join('data', model_name)
history_path = os.path.join(model_path, 'training_history.json')

with open(history_path, 'r') as j_in:
    history_data = json.load(j_in)

acc = history_data['acc']
loss = history_data['loss']
epoc = [ix for ix in range(len(acc))]

plt.plot(epoc, acc, color='salmon')
plt.xlabel('Training epoch no.')
plt.ylabel('Accuracy %')
plt.show()

plt.plot(epoc, loss, color='teal')
plt.xlabel('Training epoch no.')
plt.ylabel('Loss value')
plt.show()