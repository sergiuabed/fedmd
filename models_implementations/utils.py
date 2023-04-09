import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def plot_stats(x: list, data:list[list], save_path:str, title:str=None, x_label:str=None, y_label:str=None, data_labels:list[str]=None, x_lim:tuple=None, y_lim:tuple=None ):

    plt.figure()
    
    if title is not None:
        plt.title(title) 
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    
    if x_lim is not None:
        if len(x_lim) > 1:
            plt.xlim(x_lim[0], x_lim[1])
        else:
            plt.xlim(x_lim[0])  
    if y_lim is not None:
        if len(y_lim) > 1:
            plt.ylim(y_lim[0], y_lim[1])
        else:
            plt.ylim(y_lim[0])

    for i, y in enumerate(data):
        if len(data_labels) >= i+1:
            label = data_labels[i]
        else:
            label = None
        plt.plot(x, y, label=label)

    plt.legend()    

    plt.savefig(save_path)

    plt.close()


def save_model(model:nn.Module, path:str, epoch:int=None, accuracy:float=None, lr:float=None):

    state = {'net': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'lr': lr,
            }
    torch.save(state, path)

def load_model(net:nn.Module, path:str) -> tuple[float, int, float]:
    data = torch.load(path)
    #print(net.load_state_dict(data["net"])) 
    return data["accuracy"], data["epoch"], data["lr"]

def model_size(net:nn.Module) -> int:
    tot_size = 0
    for param in net.parameters():
        tot_size += param.size()[0]
    return tot_size

def read_stats(file_path:str) -> tuple[list]:
    with open(file_path, "r") as f:
        f.readline()
        lines = f.readlines()
    values = [line.rstrip("\n").split(",") for line in lines ]
    stats_tuples = [(int(v[0]), float(v[1]), float(v[2])) for v in values ]
    epochs, loss, acc = [list(i) for i in zip(*stats_tuples)]
    return epochs, loss, acc