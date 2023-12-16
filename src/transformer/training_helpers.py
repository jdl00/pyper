import torch
import time


def save_checkpoint(
    model_name: str, model: torch.nn.Module, optimizer: torch.optim, epoch: int
):
    # Save model state dict
    model_state_dict = model.state_dict()

    # Save optimizer state dict
    optimizer_state_dict = optimizer.state_dict()

    # Save epoch number
    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": epoch,
    }

    # Save the checkpoint
    torch.save(checkpoint, f"checkpoint_{int(time.time())}.pth")
