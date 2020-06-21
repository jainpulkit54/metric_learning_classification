import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from networks import EmbeddingNet
from networks import ClassificationNet
from loss_functions import *
from metrics import *

os.makedirs('checkpoints', exist_ok = True)

train_dataset = MNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = MNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

no_of_training_batches = len(train_loader)/batch_size
no_of_test_batches = len(test_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_classes = 10
epochs = 20

embeddingNet = EmbeddingNet()
classificationNet = ClassificationNet(embeddingNet, n_classes)

optimizer = optim.Adam(classificationNet.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.0005)

def visualize_tensor(x):
	x = x.squeeze(0)
	x = x.squeeze(0)
	x = x.numpy()
	plt.imshow(x, cmap = 'gray')
	plt.show()

def run_epoch(data_loader, model, optimizer, split = 'train', epoch_count = 0):

	model.to(device)

	if split == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0
	running_matches = 0
	samples = 0

	for batch_id, (images, targets) in enumerate(data_loader):
		
		samples = samples + images.shape[0]
		images = images.to(device)
		targets = targets.to(device)
		predictions = model(images)	
		batch_loss = cross_entropy_loss(predictions, targets)
		matches = classification_accuracy(predictions, targets)

		optimizer.zero_grad()

		if split == 'train':
			batch_loss.backward()
			optimizer.step()

		running_loss = running_loss + batch_loss.item()
		running_matches = running_matches + matches

	accuracy = running_matches/samples

	return running_loss, accuracy

def fit(train_loader, test_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss, accuracy = run_epoch(train_loader, model, optimizer, split = 'train', epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		print('Accuracy after epoch ' + str(epoch + 1) + 'is:', accuracy * 100)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, test_loader, classificationNet, optimizer = optimizer, n_epochs = epochs)