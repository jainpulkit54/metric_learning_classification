import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):

	def __init__(self):
		super(EmbeddingNet, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(32, 64, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.fc_layers = nn.Sequential(
			nn.Linear(64 * 4 * 4, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 2)
			)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.shape[0],-1)
		x = self.fc_layers(x)
		return x

	def get_embeddings(self, x):
		return self.forward(x)

class ClassificationNet(nn.Module):

	def __init__(self, embedding_net, n_classes):
		super(ClassificationNet, self).__init__()
		self.embedding_net = embedding_net
		self.n_classes = n_classes
		self.fc = nn.Linear(2, self.n_classes)
		self.non_linear = nn.PReLU()
		#self.activation = nn.LogSoftmax(dim = -1)
		
	def forward(self, x):
		x = self.non_linear(self.embedding_net(x))
		x = self.fc(x)
		#x = self.activation(self.fc(x))
		return x

	#def get_classification_scores(self, x):
	#	return self.forward(x)

	def get_embeddings(self, x):
		return self.non_linear(self.embedding_net(x))