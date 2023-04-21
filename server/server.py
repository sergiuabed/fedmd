import os
import csv
import random
import torch
from torch.utils.data import Subset, DataLoader

CLIENTS_PER_ROUND = 10
TOT_NR_CLIENTS = 100
SUBSET_BATCH_SIZE = 128

def get_architecture_clients():
    #this function returns a dictionary:
    #   key = network architecture name
    #   value = list of clients using this model architecture
    filename ="fedmd/client/client_architectures.csv"
    architecture_clients = {}
    with open(filename,'r') as data:
        for line in csv.reader(data):
            if line[0] != 'client_id':
                client_id = line[0]
                architecture = line[1]

                if architecture not in architecture_clients:
                    architecture_clients[architecture] = []
                
                architecture_clients[architecture].append(client_id)

    return architecture_clients


class Server:
    def __init__(self, clients, total_rounds, public_train_dataloader, num_samples_per_round, alpha, device):
        self.clients = clients
        self.consensus = None
        self.total_rounds = total_rounds
        self.rounds_performed = 0
        self.num_samples_per_round = num_samples_per_round
        self.alpha = alpha
        self.device = device
        self.selected_clients = None
        self.clients_scores = None

        self.public_train_dataloader = public_train_dataloader
        self.public_subset_dataloader = None

        self.accuracies = None  #stores dictionary of accuracies for each client
        self.architecture_clients = get_architecture_clients()
        self.init_accuracy_dict()
        self.choose_clients()
        self.choose_subset()

        if not os.path.isdir('logs'):   #directory for storing checkpoints when a client is revisiting its private dataset
            os.mkdir('logs')            #these logs will be deleted once the revisit is over

    #def perform_round(self):
    #    self.get_scores()


#rounds = 16

##Make sure in each round all architectures are present
##
##When drawing test_accuracy vs rounds graphs, when an architecture is used more than once during the same round, take the average of the accuracy.
##Another test_accuracy vs rounds graph idea is to plot the best performing model of each architecture and the worst performing one, according to the maximum accuracy achieved by each model of that architecture


    #def start_fedmd(self):
    #    for _ in range(self.total_rounds):
    #        self.perform_round()
    #        self.rounds_performed += 1

    def perform_round(self):

        print("Clients begin computing the scores\n")
        self.receive()
        print("Server computes consensus\n")
        self.update()
        print("Server distributes the consensus\n")
        print("Clients start digesting the consensus and training on their private data for few epochs\n")
        self.distribute()   #this will also trigger the clients to "digest" the consensus and
                            #revisit their private dataset
        print("Perform validation on every client\n")
        val_res = self.clients_validation()

        #select new clients and subset for next round
        self.choose_clients()
        self.choose_subset()

        return val_res

    def init_accuracy_dict(self):
        for c in self.clients:
            filename = os.getcwd() + f"/independent_train/client{c.client_id}/stats_{self.alpha}.csv"
            with open(filename, 'r') as data:
                highest_acc = 0
                for line in csv.reader(data):
                    if line[0] != "epoch":
                        if highest_acc < line[2]:
                            highest_acc = line[2]
                
                self.accuracies[c.client_id] = highest_acc

    def choose_clients(self):
        # makes sure every architecture type occurs in a round at least once 
        selected_clients = []
        for arch in self.architecture_clients.keys():
            c_id = random.choice(self.architecture_clients[arch])
            selected_clients.append(c_id)

        if len(selected_clients) < CLIENTS_PER_ROUND:
            remaining_clients = [str(i) for i in range(TOT_NR_CLIENTS) if str(i) not in selected_clients]
            other_clients = random.sample(remaining_clients, k=(CLIENTS_PER_ROUND - len(selected_clients)))
            selected_clients.extend(other_clients)

        #self.selected_clients = selected_clients
        self.selected_clients = [c for c in self.clients if c.client_id in selected_clients]
        print(f"Selected clients: {selected_clients}\n")

    def choose_subset(self):
        # Shuffle indexes
        tr_data_len = len(self.public_train_dataloader.dataset)
        shuffled_indexes = torch.randperm(tr_data_len)

        # Partition indexes
        train_indexes = shuffled_indexes[0: self.num_samples_per_round]

        public_subset = Subset(self.public_train_dataloader.dataset, train_indexes)
        public_subset_dataloader = DataLoader(public_subset, batch_size=SUBSET_BATCH_SIZE, num_workers=1, shuffle=False)
                                        # shuffle=False should ensure that the order
                                        # within the dataloader is consistent for
                                        # all iterations (epochs)

        self.public_subset_dataloader = public_subset_dataloader

        for c in self.selected_clients:
            c.public_train_dataloader = public_subset_dataloader

    def receive(self):
        self.clients_scores = {client.client_id: client.upload() for client in self.selected_clients}

    def update(self):
        #len_selected_clients = len(self.selected_clients)
        #self.consensus = self.clients_scores[0] / len_selected_clients
        
        nr_batches = len(self.public_subset_dataloader)
        size_batch = self.public_subset_dataloader.batch_size
        nr_classes = 100
        
        self.consensus = torch.zeros((nr_batches, size_batch, nr_classes))

        #for scores in self.clients_scores:
        #    self.consensus += scores / len_selected_clients

        selected_clients_acc = [self.accuracies[c.client_id] for c in self.selected_clients]
        sum_accs = sum(selected_clients_acc)

        for client in self.selected_clients:
            weight = self.accuracies[client.client_id] / sum_accs #this way the scores from the clients with better accuracy have more impact in the consensus
            self.consensus += self.clients_scores[client.client_id]*weight

    def distribute(self):
        for client in self.selected_clients:
            client.download(self.consensus)

    def clients_validation(self):
        val_res = {}
        for c in self.selected_clients:
            val_res[c.client_id] = c.validation_acc()
            self.accuracies[c.client_id] = val_res[c.client_id] #update with new accuracies after the end of the round
            print(f"Client {c.client_id} accuracy: {val_res[c.client_id]}\n")

        return val_res

        