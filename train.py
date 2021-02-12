import time
import os
import typing

import torch
from torch import nn, autograd, optim
from torch.utils import data
from torchvision import transforms
import numpy as np

import constant
import dataset
import model


# MARK: - Constants
# Train
LEARNING_RATE = 1e-4
N_EPOCHS: typing.Final = 200
TRAIN_BATCH_SIZE: typing.Final = 300
TRAIN_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.RandomCrop(constant.IMAGE_HEIGHT_WIDTH, padding=(constant.IMAGE_HEIGHT_WIDTH // 8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Validation/test
VAL_BATCH_SIZE: typing.Final = TRAIN_BATCH_SIZE
VAL_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
VAL_EPOCH_INTERVAL: typing = 1    ; """Validate every 10 epochs."""


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


def main():
    # MARK: Variables
    network = model.DenseNet762()
    network = network.cuda()

    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    train_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_TRAIN_SUB_DIR,
        download=True,
        transform=TRAIN_TRANSFORMS
    )
    # We need to drop the last mini-batch because we have batch normalizations in our network.
    # If the last batch size is 1, batch norm won't work.
    train_dataloader = data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    validation_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_VALIDATION_SUB_DIR,
        download=True,
        transform=VAL_TRANSFORMS
    )
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, num_workers=4)

    for epoch in range(1, N_EPOCHS + 1):
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

        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        print(f"Epoch {epoch}: accuracy: {train_correct_count}/{train_count} ({train_correct_count / train_count}), total loss: {train_loss}, average loss: {train_loss / train_count}, time: {train_time} seconds")
        del image, label, prediction, loss, max_prediction_indices

        # MARK: Validate
        if (epoch % VAL_EPOCH_INTERVAL == 0):
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

            print(f"Validation: accuracy: {validation_correct_count}/{validation_count} ({validation_correct_count / validation_count}), total loss: {validation_loss}, average loss: {validation_loss / validation_count}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
