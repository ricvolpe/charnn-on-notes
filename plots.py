import matplotlib.pyplot as plt
import os
import json

model_name = 'notes_network_220819_1710'

model_path = os.path.join('data', model_name)
history_path = os.path.join(model_path, 'training_history.json')

with open(history_path, 'r') as j_in:
    history_data = json.load(j_in)

acc = history_data['acc']
loss = history_data['loss']
if 'val_acc' in history_data.keys():
    val_acc = history_data['val_acc']
    val_loss = history_data['val_loss']
epoc = [ix for ix in range(len(acc))]

plt.plot(epoc, acc, label='training')
if val_acc:
    plt.plot(epoc, val_acc, label='validation')
plt.xlabel('Training epoch no.')
plt.ylabel('Accuracy %')
plt.legend()
plt.show()

plt.plot(epoc, loss, label='training')
if val_loss:
    plt.plot(epoc, val_loss, label='validation')
plt.xlabel('Training epoch no.')
plt.ylabel('Loss value')
plt.legend()
plt.show()