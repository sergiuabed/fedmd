import os, sys
#sys.path.append('models_implementations')
import torch
import torchvision
import torchvision.transforms as transforms
from models_implementations.train_on_cifar import _data_processing
from torchvision.datasets import CIFAR10
from data_utils import read_data_splits
from client.private_dataloader import ClientPrivateDataset
from torch.utils.data import DataLoader
from client.client import Client
import matplotlib.pyplot as plt

BATCH_SIZE = 128
NUM_WORKERS = 4

PRIVATE_TRAIN_DATA_DIR = os.path.join('.', 'data', 'cifar100', 'data', 'train') # location of json files storing the data splits
PRIVATE_TEST_DATA_DIR = os.path.join('.', 'data', 'cifar100', 'data', 'test')
ALPHA = 0.00    # non-IID

def main():

    # MAKE SURE TO RUN "setup_datasets.sh" SCRIPT BEFORE ANYTHING!

    public_train_dataloader, public_validation_dataloader, public_test_dataloader = _data_processing(CIFAR10)
    
    client_ids, train_data, test_data = read_data_splits(PRIVATE_TRAIN_DATA_DIR, PRIVATE_TEST_DATA_DIR, ALPHA)

    # train_clients: list of client ids 
    # train_data: dictionary with key=client_id and value=(dictionary storing the data of the client)
    # test_data: dictionary storing the data for validation. It is not a dictionary of dictionaries. It is used by all clients
    
#    clients = []
#    for u in client_ids:
#        c_traindata = ClientPrivateDataset(train_data[u], train=True)
#        c_testdata = ClientPrivateDataset(test_data, train=False)   # the same test dataset for all clients
#
#        private_train_dataloader = DataLoader(c_traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
#        private_test_dataloader = DataLoader(c_testdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#    
#        clients.append(Client(u, public_train_dataloader, public_validation_dataloader, private_train_dataloader, 
#                              private_test_dataloader, None, model))


    c_traindata = ClientPrivateDataset(train_data["0"], train=True)
    c_testdata = ClientPrivateDataset(test_data, train=False)   # the same test dataset for all clients

    print()
    print()
    print("length=%d"%len(c_traindata))
    print(c_traindata)
    print()
    print()

    private_train_dataloader = DataLoader(c_traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    private_test_dataloader = DataLoader(c_testdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


if __name__ == "__main__":
    print("ciao")
    main()




















    # define public dataloader and private dataloader
        # public dataloader (CIFAR10): use torchvision implementation

        # private dataloader (CIFAR100): use assistant's data splits for each user
            # in setup_datasets.sh delete the code related to CIFAR10 and run it to load the dataset
            # copy "create_client()" and "setup_client()" functions
            
    # define client class and server class
    
    # client class:
        # pass the model pre-trained on cifar10 (public dataset)
        # replace the last fc layer with a fc layer for cifar100
        # train the model on the private dataset (i.e. the client's subset of cifar100)
        # 
    # write jupyter notebook to execute FedMD