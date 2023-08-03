from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use("ggplot")


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, args):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f"model_checkpoints/best_model_{args.model_name}_{args.num_classes}.pth",
            )


def save_model(epochs, model, optimizer, args):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    curr_dt = str(datetime.now())

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"model_checkpoints/final_model_{args.model_name}_{args.num_classes}_{curr_dt}.pth",
    )


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    curr_dt = str(datetime.now())
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"results/plots/accuracy_{curr_dt}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/plots/loss_{curr_dt}.png")
