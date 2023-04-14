import numpy as np
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fedmd.models_implementations.train_on_cifar import _training, _validation
import os

LOCAL_EPOCH = 5
LR = 0.1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

FILE_PATH = os.getcwd() + '/logs'   # path for storing model checkpoints and logs on intermediary states of the model

def initialize_scores(len_public_dataloader, output_dim, device):
    return torch.ones((len_public_dataloader, output_dim)).to(device) * float("inf")

class Client:

    def __init__(self, client_id, public_train_dataloader,private_train_dataloader,
                 private_validation_dataloader, current_consensus, model, model_architecture, device=None):
        self._model = model
        self.model_architecture = model_architecture
        self.client_id = client_id
        self.device = device

        self.public_train_dataloader = public_train_dataloader
        self.private_train_dataloader = private_train_dataloader
        self.private_validation_dataloader = private_validation_dataloader

        self.current_local_scores = initialize_scores(len(public_train_dataloader), 100, device)
        self.current_consensus = current_consensus

        self.consensus_loss_func = nn.L1Loss()
        self.consensus_optimizer = optim.Adam(self._model.parameters(), 0.001)  # optimizer suggested in FedMD paper with starting lr=0.001

        self.accuracies = []
        self.losses = []

    def upload(self):
        print(f"Client {self.client_id} starts computing scores.\n")
        self._model.to(self.device)
        for data in self.public_train_dataloader:
            idx = data[1]
            x = data[0]
            x = x.to(self.device)

            self.current_local_scores[idx, :] = self._model(x).detach()

        return self.current_local_scores

    def download(self, current_consensus): #calling this also triggers digest and revisit(i.e. private_train)
        self.current_consensus = current_consensus
        print(f"Client {self.client_id} starts digest phase\n")
        self.digest()
        print(f"Client {self.client_id} revisits its private data for {LOCAL_EPOCH} epochs\n")
        self.private_train()

    def private_train(self):
        _training(
            self.net, self.private_train_dataloader, self.private_validation_dataloader, LOCAL_EPOCH , LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
        )

        os.remove(FILE_PATH + "/best_model.pth")
        os.remove(FILE_PATH + "/stats.csv")

    def digest(self):   # i.e. approach consensus
        running_loss = 0

        self._model.to(self.device)
        for data in self.public_train_dataloader:
            idx = data[1]
            x = data[0].to(self.device)
            y_consensus = self.current_consensus[idx, :].to(self.device)
            self.consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            self.consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss

    def validation_acc(self):
        acc = _validation(self._model, self.private_validation_dataloader)
        return acc