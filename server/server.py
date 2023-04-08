class Server:
    def __init__(self, clients):
        self.clients = clients
        self.current_consensus = None

    #def perform_round(self):
    #    self.get_scores()

    def perform_round(self):
        self.receive()
        self.update()
        self.distribute()

    def receive(self):
        self.clients_scores = [client.upload() for client in self.clients]

    def update(self):
        len_clients = len(self.clients)
        self.consensus = self.clients_scores[0] / len_clients
        for scores in self.clients_scores[1:]:
            self.consensus += scores / len_clients

    def distribute(self):
        """Distribute the scores of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)
