import time
import os
import typing
import argparse
import warnings

import torch
from torch import nn, autograd, optim
from torch.utils import data
from torchvision import transforms
import numpy as np

import constant
import dataset
import densenet
import epoch_logger


# MARK: - Constants
# Train
LEARNING_RATE = 0.06
TRAIN_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.RandomCrop(constant.IMAGE_HEIGHT_WIDTH, padding=(constant.IMAGE_HEIGHT_WIDTH // 8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Validation/test
VAL_BATCH_SIZE: typing.Final = 300
VAL_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
VAL_EPOCH_INTERVAL: typing.Final = 1    ; """Validation epoch interval."""

# Save
CHECKPOINT_SAVE_EPOCH_INTERVAL: typing.Final = 25    ; """Checkpoints save interval."""


# MARK: - Helpers
def calculate_val_accuracy(network: nn.Module, val_loader: data.DataLoader, is_gpu=True):
    """ Util function to calculate val set accuracy,
    both overall and per class accuracy
    Args:
        val_loader (torch.utils.data.DataLoader): val set
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """
    warnings.warn("The original `calculate_val_accuracy` function is no longer used because it's highly unoptimized.", DeprecationWarning)

    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for i in range(constant.N_CLASSES))
    class_total = list(0. for i in range(constant.N_CLASSES))

    for data in val_loader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = network(autograd.Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100*correct/total, class_accuracy


def main(train_batch_size: int, n_epochs: int, checkpoint_save_dir: str, train_log_filename: str, validation_log_filename: str):
    # MARK: Verify parameters
    if (not os.path.exists(checkpoint_save_dir)):
        os.makedirs(checkpoint_save_dir)
        print(f"Created checkpoint save dir `{checkpoint_save_dir}`.")
    elif (os.path.isdir(checkpoint_save_dir)):
        print(f"Using existing checkpoint save dir `{checkpoint_save_dir}`.")
        existing_filenames = os.listdir(checkpoint_save_dir)
        if existing_filenames:
            print(f"Existing checkpoint files in this directory will be overwritten.")
    else:
        raise FileExistsError(f"Checkpoint save dir `{checkpoint_save_dir}` is not a folder.")

    # MARK: Variables
    network = densenet.DenseNet762()
    network = network.cuda()

    # optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_function = nn.CrossEntropyLoss()

    train_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_TRAIN_SUB_DIR,
        download=True,
        transform=TRAIN_TRANSFORMS
    )
    # We need to drop the last mini-batch because we have batch normalizations in our network.
    # If the last batch size is 1, batch norm won't work.
    train_dataloader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, drop_last=True)

    validation_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_VALIDATION_SUB_DIR,
        download=True,
        transform=VAL_TRANSFORMS
    )
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, num_workers=2)

    for epoch in range(1, n_epochs + 1):
        # MARK: Train
        train_start_time = time.time()

        network.train()

        train_loss = 0.
        train_count = 0
        train_correct_count = 0
        for image, label in train_dataloader:
            image = image.cuda()
            label = label.cuda()

            prediction = network(image)
            loss = loss_function(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += image.shape[0]

            _, max_prediction_indices = torch.max(prediction, -1)
            train_correct_count += torch.sum(max_prediction_indices == label).item()

        scheduler.step()

        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        print(f"Epoch {epoch}: accuracy: {train_correct_count}/{train_count} ({train_correct_count / train_count}), total loss: {train_loss}, average loss: {train_loss / train_count}, time: {train_time} seconds")
        epoch_logger.log_epoch_details_to_file(epoch, train_count, train_correct_count, (train_correct_count / train_count), train_loss, (train_loss / train_count), train_time, train_log_filename)

        # MARK: Validate
        if (epoch % VAL_EPOCH_INTERVAL == 0):
            validation_start_time = time.time()

            network.eval()

            validation_loss = 0.
            validation_count = 0
            validation_correct_count = 0

            with torch.no_grad():    # [Very important] Reduce memory usage.
                for image, label in validation_dataloader:
                    image = image.cuda()
                    label = label.cuda()

                    prediction = network(image)
                    loss = loss_function(prediction, label)
                    validation_loss += loss.item()
                    validation_count += image.shape[0]

                    _, max_prediction_indices = torch.max(prediction, -1)
                    validation_correct_count += torch.sum(max_prediction_indices == label).item()

            validation_end_time = time.time()
            validation_time = validation_end_time - validation_start_time
            print(f"Validation: accuracy: {validation_correct_count}/{validation_count} ({validation_correct_count / validation_count}), total loss: {validation_loss}, average loss: {validation_loss / validation_count}, time: {validation_time} seconds")
            epoch_logger.log_epoch_details_to_file(epoch, validation_count, validation_correct_count, (validation_correct_count / validation_count), validation_loss, (validation_loss / validation_count), validation_time, validation_log_filename)

        # MARK: Save checkpoint
        if ((epoch % CHECKPOINT_SAVE_EPOCH_INTERVAL) == 0):
            checkpoint_filename = os.path.join(checkpoint_save_dir, f"{epoch}.pth")
            torch.save(network.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved as `{checkpoint_filename}`")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--checkpoint_save_dir", type=str, default="checkpoints")
    parser.add_argument("--train_log_filename", type=str, default="checkpoints/train_log.csv")
    parser.add_argument("--validation_log_filename", type=str, default="checkpoints/validation_log.csv")
    args = parser.parse_args()

    main(args.train_batch_size, args.n_epochs, args.checkpoint_save_dir, args.train_log_filename, args.validation_log_filename)
