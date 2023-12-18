import torch
from torch import nn
from torch.nn import optim
from torch.utils.data import Dataset


from .model import Transformer, summary
from .args.train_args import TTrainArgs
from .args.model_args import TModelArgs
from .training_helpers import save_checkpoint


def train_expert(
    model_name: str,
    model_args: TModelArgs,
    train_args: TTrainArgs,
    train_dataloader: Dataset = None,
    test_dataloader: Dataset = None,
    max_batch_size: int = 16,
):
    model = Transformer(model_args)
    summary(model)

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
    optimizer = optim.AdamW(model.parameters(), lr=train_args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=train_args.step_size, gamma=train_args.gamma
    )
    gradient_accumulation_steps = max(1, max_batch_size // train_args.batch_size)

    # Actual training loop
    for epoch in range(train_args.epochs):
        model.zero_grad()  # Reset gradients at the start of each epoch
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss.backward()
            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

        scheduler.step()

        save_checkpoint(
            model_name=model_name, model=model, optimizer=optimizer, epoch=epoch
        )
