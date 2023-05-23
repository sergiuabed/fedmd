# FedMD

Contributors:

- Sergiu Abed
- Riccardo Musumarra

## 1. Introduction

Federated learning is a machine learning technique in which multiple models collaborate by sharing their knowledge so that each model can improve its performance when executing a certain task.

The need for a framework like federated learning arises from the low availability of training data of a client and the privacy issues that can be caused by collecting sensitive data from multiple clients in a centralized fashion. Instead of sharing their own data, each client updates the global model received from the coordinating server on its own data and sends the updated model to the server which then averages the received models from all clients to obtain a new global model [1].

There are 3 main challanges that federated learning faces:

- statistical heterogeneity
- system heterogeneity
- privacy threats

FedMD addresses the first two.

The initial federated learning implementation (as described above) works only when all clients have the same network architecture for their models, since model parameters averaging doesn't work otherwise. It makes perfect sense to think that each client would want to have their own implementation of their model based on their computational resources. Some institutions may have more computational power at their disposal which they can use to train better performing models, while others may have limited resources and so they would have to resort to more computational efficient architectures. One solution to this problem (i.e. system heterogeneity) is FedMD (Federated Model Distillation) [2].

## 2. Related work

The goal of this work is reproducing the experiments presented in the paper "FedMD: Heterogeneous Federated Learning via Model Distillation" [2]. However, our problem setting is a bit different than what the authors simulated. We address the difference in the "Experiment" section of this text.

The paper presents a federated setting consisting of 10 clients under i.i.d. (independent and identically distributed) and non-i.i.d. datasets conditions and each client can choose its own model architecture. For one of the experiments, they used CIFAR10 as public dataset and CIFAR100 as private dataset. Each client has been assigned a subset of CIFAR100. In the i.i.d. case, the task of a client is to classify each image as one of the 100 possible subclasses, whereas in the non-i.i.d. situation, a client must classify each image as belonging to one of the 20 possible superclasses (a superclass has a larger granularity then a subclass, i.e. multiple subclasses may fall under the same superclass).

## 3. Methodology

In FedMD setting, we no longer have a global model that is being shared among all clients. Instead, the coordinating server picks a certain number of clients per round and a certain dataset which the chosen clients must use for that round for collaborative training.

We have two types of datasets: a public dataset and a private dataset. The public dataset is used for transfer learning, i.e. each client first trains its model on this common dataset, and then each client trains the model on its private dataset.

Once training on both public and private datasets is finished, the clients can start getting involved in collaboration rounds.

A round consists of the following phases:

- public subset and clients selection: the server chooses a subset of the public dataset (or the whole dataset) and the clients that must participate in the round and sends to each of them the chosen subset

- communicate: each client computes the scores on each image of the subset and sends them to the server

- aggregate: the server averages the scores received from the clients to compute a consensus

- distribute: the consensus is sent to all the participating clients

- digest: the clients perform training on the received consensus, i.e. for each image in the public subset the loss is computed between the output of the model and the consensus value corresponding to that image (i.e. in the loss function, instead of giving as input the label of the image, you put the average of the outputs for that image from all the participating clients)

- revisit: each client trains again on its own private dataset for few epochs

For optimzation, we used Adam for the digest phase (as suggested by the authors of paper [2]) and SGD for the revisit phase.

## 4. Experiment

 In the following we describe our problem setting:

- number of clients: 100
- number of clients participating in a round: 10
- public dataset: CIFAR10
- private dataset: CIFAR100 split in 100 subsets splits using Dirichlet's distribution to obtain i.i.d. or non-i.i.d. case. We used the splits provided by our teaching assistant
- network architectures:
  - ResNet20-BN (batch normalization)
  - ResNet20-GN (group normalization)
  - DenseNet20
  - DenseNet10
  - ShuffleNetBig
  - ShuffleNetSmall
- digest phase optimizer: Adam with LR (learing rate) set to 0.001
- revisit phase optimizer: SGD with LR=0.001 for i.i.d. case and LR=0.0001 for non-i.i.d.

 The task of a client is to classify each image as one of the possible 100 subclasses in both i.i.d. and non-i.i.d conditions.

### 4.1 Non-i.i.d. results
