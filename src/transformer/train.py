import torch
from torch import nn
from torch.nn import optim
from torch.utils.data import Dataset


from .model import Transformer
from .args.t_train_args import TTrainArgs
from .training_helpers import save_checkpoint


def create_parameters(model: nn.Module, train_args: TTrainArgs):
    # Declare the optimiser
    optimizer = optim.AdamW(model.parameters(), lr=train_args.lr)

    # declare the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create the cosine learning rate annealer
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
    )

    return (optimizer, loss_fn, lr_scheduler)


def train_expert(
    model: Transformer,
    model_name: str,
    train_args: TTrainArgs,
    train_dataloader: Dataset = None,
    test_dataloader: Dataset = None,
    max_batch_size: int = 16,
):
    # Check available devices and set the model to the correct device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Move the model to the selected device
    model.to(device)

    # Create the optimisers loss function and schedulers
    optimizer, loss, scheduler = create_parameters(model, train_args)

    gradient_accumulation_steps = max(1, max_batch_size // train_args.batch_size)

    # Actual training loop
    for epoch in range(train_args.epochs):
        model.zero_grad()  # Reset gradients at the start of each epoch
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss(outputs, targets)

            loss.backward()
            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

        scheduler.step()

        save_checkpoint(
            model_name=model_name, model=model, optimizer=optimizer, epoch=epoch
        )
