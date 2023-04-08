import numpy as np
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models_implementations.train_on_cifar import _training
import os

LOCAL_EPOCH = 2
LR = 0.1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

FILE_PATH = os.getcwd() + '/logs'   # path for storing model checkpoints and logs on intermediary states of the model

class Client:

    def __init__(self, client_id, public_train_dataloader, public_validation_dataloader,
                 private_train_dataloader, private_validation_dataloader, current_consensus, model, device=None):
        self._model = model
        self.id = client_id
        #self.train_data = train_data
        #self.eval_data = eval_data
        #self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) if self.train_data.__len__() != 0 else None
        #self.testloader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) if self.eval_data.__len__() != 0 else None
        self.device = device
        #self.batch_size = batch_size

        self.public_train_dataloader = public_train_dataloader
        self.private_train_dataloader = private_train_dataloader
        self.public_validation_dataloader = public_validation_dataloader
        #might not need private_validation_dataloader
        self.private_validation_dataloader = private_validation_dataloader

        self.current_local_scores = None
        self.current_consensus = current_consensus
        self.cifar_mode = 10    # either 10 or 100

        self.consensus_loss_func = nn.L1Loss()
        self.consensus_optimizer = self._model.parameters()

        self.accuracies = []
        self.losses = []

        self.change_fc_layer()

    def change_fc_layer(self):
        cifar100_fc = nn.Linear(in_features=self._model.fc.in_features, out_features=100)
        self._model.fc = cifar100_fc

    def upload(self):
        for data in self.public_train_dataloader:
            idx = data[0]
            x = data[1]
            x = x.to(self.device)
            self.current_local_scores[idx, :] = self(x).detach()

        return self.current_local_scores

    def download(self, current_consensus):
        self.current_consensus = current_consensus

#    def public_train(self):
#        
#        #SWITCH LAST LAYER FROM FC WITH OUT_SIZE=100 TO OUT_SIZE=10
#        #THIS IS BECAUSE THE PUBLIC DATASET IS CIFAR10 AND PRIVATE DATASET IS CIFAR100
#
#        return _training(
#            self.net, self.public_train_dataloader, self.public_validation_dataloader, LOCAL_EPOCH, LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
#        )
#    
#    def private_train(self):
#
#        #SWITCH LAST LAYER FROM FC WITH OUT_SIZE=10 TO OUT_SIZE=100
#
#        return _training(
#            self.net, self.private_train_dataloader, self.private_validation_dataloader, LOCAL_EPOCH, LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
#        )

#    def local_train(self, public):  # 'public' indicates on which dataset to train (i.e. public or private)
#        if public is True:
#            train_dataloader = self.public_train_dataloader
#            validation_dataloader = self.public_validation_dataloader
#
#            if self.cifar_mode != 10:
#                # the last layer must be replaced with a fc layer that outputs
#                # 10 values when training on cifar10 (analogous for cifar100)
#                self.change_fc_layer(10)
#
#        else:
#            train_dataloader = self.private_train_dataloader
#            validation_dataloader = self.private_validation_dataloader
#
#            if self.cifar_mode != 100:
#                self.change_fc_layer(100)

    def private_train(self, revisit):
        # revisit should be false only when the model is trained on the
        # private dataset for the first time
        losses, accuracies = _training(
            self.net, self.private_train_dataloader, self.private_validation_dataloader, LOCAL_EPOCH if revisit is True else 30, LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
        )

        self.losses.extend(losses)
        self.accuracies.extend(accuracies)


        #    return _training(
        #        self.net, self.public_train_dataloader, self.public_validation_dataloader, LOCAL_EPOCH, LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
        #    )
        #else:
        #    return _training(
        #        self.net, self.private_train_dataloader, self.private_validation_dataloader, LOCAL_EPOCH, LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
        #    )

    def digest(self):   # i.e. approach consensus
        running_loss = 0

        for data in self.public_training_dataloader:
            idx = data[0]
            x = data[1].to(self.device)
            y_consensus = self.current_consensus[idx, :].to(self.device)
            self.consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            self.consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss
