import torch
import torch.nn as nn

def cross_entropy_loss(predictions, targets):
	
	#loss_fn = nn.NLLLoss(reduction = 'mean')
	loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
	loss = loss_fn(predictions, targets)
	return loss
