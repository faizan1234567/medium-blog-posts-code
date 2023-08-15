import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random
import math
from model import SimpleNN
from utils import log_image_table
from test import validate
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type = int, default=3, help= "wandb experiments")
    parser.add_argument('--epochs', type = int, default= 10, help = "number of epochs")
    parser.add_argument('--batch', type = int, default=128, help="batch size")
    parser.add_argument('--lr', type = float, default=1e-3, help="learning rate")
    return parser.parse_args()

# create a data loader function
def create_dataloader(is_train, batch_size, slice=5):
    """
    Get a training dataloader for loading processed batched dataset for training
    Args:
      is_train: bool
      batch_size: int
      slice: int

    return:
      loader: torch.utils.data.DataLoader
    """
    full_dataset = torchvision.datasets.FashionMNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(dataset=sub_dataset,
                                         batch_size=batch_size,
                                         shuffle=True if is_train else False,
                                         pin_memory=True, num_workers=2)
    return loader


# choose cuda if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# train for specified number of experiment with given epochs
def train(experiments, epochs = 10, batch_size = 128 , lr = 1e-3):
    for _ in range(experiments):
    # initialise a wandb run
        wandb.init(
            project="Fashion-MNIST-Classification",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dropout": random.uniform(0.01, 0.80),
                })

        # Copy your config
        config = wandb.config

        # Get the data
        train_dl = create_dataloader(is_train=True, batch_size=config.batch_size)
        valid_dl = create_dataloader(is_train=False, batch_size=2*config.batch_size)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        # A simple MLP model
        model = SimpleNN(in_channels=1, dropout=config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                example_ct += len(images)
                metrics = {"train/train_loss": train_loss,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/example_ct": example_ct}

                if step + 1 < n_steps_per_epoch:
                    # Log train metrics to wandb
                    wandb.log(metrics)

                step_ct += 1

            val_loss, accuracy = validate(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

            # log train and validation metrics to wandb
            val_metrics = {"val/val_loss": val_loss,
                        "val/val_accuracy": accuracy}
            wandb.log({**metrics, **val_metrics})

            print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # If you had a test set, this is how you could log it as a Summary metric
    wandb.summary['test_accuracy'] = 0.8

    # Close your wandb run
    wandb.finish()


if __name__ == "__main__":
    args = read_args()
    # train now
    if args.experiments and args.epochs:
        train(args.experiments, args.epochs)
    else:
        train(3, 10)


