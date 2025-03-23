import sys
sys.path.append('Soft_Actor-Critic-implementation/')
import yaml
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def check_model_device(model):
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

def print_model_parameters(model):
    print("Weights:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

def check_gradients(model):
    print("Gradients:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.grad}")

def print_weight_updates(model, prev_params):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Difference in {name}: {torch.sum(param.data - prev_params[name])}")


def save_checkpoint(epoch, loss, model, optimizer):
    PATH = "../../checkpoint.pt"

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)

def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, eval_data = train_test_split(test_data, test_size=0.5, random_state=42)

    return train_data, eval_data, test_data
