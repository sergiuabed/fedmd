import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from typing import Tuple

from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from .utils import save_model, load_model
#from utils import save_model, load_model, model_size



DEVICE = "cuda"
#DEVICE = "cpu"
DATA_PATH = os.getcwd()

BATCH_SIZE = 128
#BATCH_SIZE = 8
VAL_RATIO = 0.2  # Fraction of the training set used for validation
NUM_WORKERS = 4


def _data_processing(dataset: CIFAR10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Define transforms for training phase
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Normalizes tensor with mean and standard deviation
        ]
    )
    
    # Define transforms for the evaluation phase
    eval_transform = train_transform

    # Prepare Pytorch train/test Datasets
    train_dataset = dataset(
        root=DATA_PATH, train=True, transform=train_transform, download=True
    )
    test_dataset = dataset(root=DATA_PATH, train=False,
                           transform=eval_transform)

    # Get training dataset length
    tr_data_len = len(train_dataset)

    # Shuffle indexes
    shuffled_indexes = torch.randperm(tr_data_len)

    # Partition indexes
    # train_indexes = shuffled_indexes[0:100]
    # val_indexes = shuffled_indexes[100:200]
    #test_dataset = Subset(test_dataset, [i for i in range(200)])
    train_indexes = shuffled_indexes[0: int(tr_data_len * (1 - VAL_RATIO))]
    val_indexes = shuffled_indexes[int(tr_data_len * (1 - VAL_RATIO)): tr_data_len]

    tr_dataset = Subset(train_dataset, train_indexes)
    val_dataset = Subset(train_dataset, val_indexes)

    # Check dataset sizes
    print('Train Dataset: {}'.format(len(tr_dataset)))
    print('Valid Dataset: {}'.format(len(val_dataset)))
    print('Test Dataset: {}'.format(len(test_dataset)))

    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    train_dataloader = DataLoader(
        tr_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_dataloader, val_dataloader, test_dataloader


def _training(
    net: torch.nn.Module,
    tr_set: DataLoader,
    val_set: DataLoader,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    file_path:str,
) -> None:
        
    # Load checkpoint if present
    if os.path.isfile(file_path + "/checkpoint.pth"):
        _, epoch, lr = load_model(net, file_path + "/checkpoint.pth")
        cur_epoch = epoch
    else:
        cur_epoch = 0
        # Delete previous stats file
        if os.path.isfile(file_path + "/stats.csv"):
            os.remove(file_path + "/stats.csv")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Choose parameters to optimize
    #parameters_to_optimize = net.parameters()
    parameters_to_optimize = [p for p in net.parameters() if p.requires_grad]

    # Define optimizer
    optimizer = optim.SGD(
        parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # Define scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs-cur_epoch, eta_min=1e-2
    )
    
    # Send to device
    net = net.to(DEVICE)
    # Optimize
    cudnn.benchmark 

    # Train
    current_step = 0
    max_accuracy = 0

    #accuracies = []
    #losses = []

    for epoch in range(cur_epoch, num_epochs):
        print(
            "Starting epoch {}/{}, LR = {}".format(
                epoch + 1, num_epochs, scheduler.get_lr()
            )
        )
        sum_losses = torch.zeros(1).to(DEVICE)

        # Iterate over the training dataset in batches
        for images, labels in tr_set:
            # Bring data over the device of choice
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            net.train()  # Sets module in training mode

            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            outputs = net(images)

            # Compute loss based on output and ground truth
            loss = criterion(outputs, labels)
            sum_losses += loss

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # Step the scheduler
        scheduler.step()

        # Compute and log the average loss over all batches
        avg_loss = sum_losses.item() / len(tr_set)
        #losses.append(avg_loss)
        print(f"Current Avg Loss = {avg_loss}")

        # Compute validation accuracy
        acc = _validation(net, val_set)
        #accuracies.append(acc)
        print(f"Current Val Accuracy = {acc}")
        
        # Save the best model
        if acc > max_accuracy:
            save_model(net, file_path + "/best_model.pth", epoch, acc, scheduler.get_last_lr()[-1])
            max_accuracy = acc
        # Checkpoint
        save_model(net, file_path + "/checkpoint.pth", epoch, acc, scheduler.get_last_lr()[-1])

        # Record stats
        with open(file_path + "/stats.csv", "a") as f:
            if epoch == 0:
                f.write("epoch,avg_loss,accuracy\n")
            f.write(f"{epoch},{avg_loss},{acc}\n")

    print("Max Validation Accuracy: {}".format(max_accuracy))

    # Delete checkpoint
    os.remove(file_path + "/checkpoint.pth")

    #return losses, accuracies

def _validation(net: torch.nn.Module, val_set: DataLoader):
    # This will bring the network to GPU if DEVICE is cuda
    net = net.to(DEVICE)
    net.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in val_set:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / (len(val_set)*BATCH_SIZE)

    net.train(True)  # Set network to training mode

    return accuracy


def _testing(net: nn.Module, test_set: DataLoader, model_path: str=None):
    # Load model if available
    if model_path is not None:
        load_model(net,model_path)

    net = net.to(DEVICE)
    net.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in test_set:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / (len(test_set)*BATCH_SIZE)

    # Save model with test accuracy if path available
    if model_path is not None:
        save_model(net, model_path, None, accuracy, None)

    print("Test Accuracy: {}".format(accuracy))


def train_on_cifar(
    net: torch.nn.Module,
    num_epochs: int,
    lr: float = 1e-1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    cifar_100: bool = True,
    file_path: str = os.getcwd(),
) -> None:
    if cifar_100:
        dataset = CIFAR100
    else:
        dataset = CIFAR10

    training_set, validation_set, test_set = _data_processing(dataset)

    _training(
        net,
        training_set,
        validation_set,
        num_epochs,
        lr,
        momentum,
        weight_decay,
        file_path,
    )

    _testing(net, test_set, file_path + "/best_model.pth")
    
# if __name__ == "__main__":
#     from shufflenetv2 import ShuffleNetV2
    
#     LR = 1e-1     
#     MOMENTUM = 0.9       
#     WEIGHT_DECAY = 1e-4  
#     NUM_EPOCHS = 5
#     FILE_PATH =  os.getcwd()
#     NUM_CLASSES = 100
#     GROUPS = 2
#     REPEATS = [2, 4, 2]
#     OUT_CHANNELS = [16, 32, 64, 128, 256]

#     net = ShuffleNetV2(REPEATS, OUT_CHANNELS, NUM_CLASSES)
#     print(f"Num Parameters = {model_size(net)}")
#     train_on_cifar(net, NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, cifar_100=True, file_path=FILE_PATH)
