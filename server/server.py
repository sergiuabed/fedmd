import os
import csv
import random

CLIENTS_PER_ROUND = 10
TOT_NR_CLIENTS = 100

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
    def __init__(self, clients, total_rounds, device):
        self.clients = clients
        self.consensus = None
        self.total_rounds = total_rounds
        self.rounds_performed = 0
        self.device = device
        self.selected_clients = None
        self.clients_scores = None

        self.architecture_clients = get_architecture_clients()
        self.choose_clients()

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
        self.receive()
        self.update()
        self.distribute()   #this will also trigger the clients to "digest" the consensus and
                            #revisit their private dataset
        val_res = self.clients_validation()

        #select new clients for next round
        selected_clients = self.choose_clients()

        return val_res

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
        return selected_clients

    def receive(self):
        self.clients_scores = [client.upload() for client in self.selected_clients]

    def update(self):
        len_selected_clients = len(self.selected_clients)
        self.consensus = self.clients_scores[0] / len_selected_clients
        for scores in self.clients_scores[1:]:
            self.consensus += scores / len_selected_clients

    def distribute(self):
        for client in self.selected_clients:
            client.download(self.consensus)

    def clients_validation(self):
        val_res = {}
        for c in self.selected_clients:
            val_res[c.client_id] = c.validation_acc()
        
        return val_res

        