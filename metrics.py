import numpy as np
import torch.nn as nn

softmax = nn.Softmax(dim = -1)

def classification_accuracy(predictions, targets):

	predictions = softmax(predictions)
	predictions = predictions.detach().cpu().numpy()
	targets = targets.detach().cpu().numpy()
	indices = np.argmax(predictions, axis = 1)
	matches = 0

	for i in range(len(targets)):
		if indices[i] == targets[i]:
			matches = matches + 1

	return matches
