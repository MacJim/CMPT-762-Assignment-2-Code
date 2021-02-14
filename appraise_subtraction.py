import os
import typing

import torch
from torch.utils import data
import torchvision.datasets

from densenet import DenseNet762
from infer import TEST_TRANSFORMS
import dataset
import constant


DATASET_ROOT_DIR = "/tmp/cifar100"
BATCH_SIZE = 2000


def _get_correct_count_of_dataloader(checkpoint_filename: str, dataloader: data.DataLoader) -> typing.Tuple[int, int]:
    total_count = 0
    correct_count = 0

    with torch.no_grad():
        network = DenseNet762()
        network = network.cuda()
        network.load_state_dict(torch.load(checkpoint_filename))
        network.eval()

        for image, label in dataloader:
            image = image.cuda()
            label = label.cuda()

            prediction = network(image)

            total_count += image.shape[0]

            _, max_prediction_indices = torch.max(prediction, -1)
            correct_count += torch.sum(max_prediction_indices == label).item()

    return (correct_count, total_count)


def get_total_correct_count(checkpoint_filename: str) -> typing.Tuple[int, int]:
    # 50000 images
    train_dataset = torchvision.datasets.CIFAR100(
        root=DATASET_ROOT_DIR,
        train=True,
        download=True,
        transform=TEST_TRANSFORMS,
    )
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 10000 images
    test_dataset = torchvision.datasets.CIFAR100(
        root=DATASET_ROOT_DIR,
        train=False,
        download=True,
        transform=TEST_TRANSFORMS,
    )
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    train_correct, train_total = _get_correct_count_of_dataloader(checkpoint_filename, train_dataloader)
    test_correct, test_total = _get_correct_count_of_dataloader(checkpoint_filename, test_dataloader)

    correct = train_correct + test_correct
    total = train_total + test_total

    # print(f"Train accuracy: {train_correct}/{train_total} ({train_correct / train_total})")
    # print(f"Test accuracy: {test_correct}/{test_total} ({test_correct / test_total})")
    # print(f"Overall accuracy: {correct}/{total} ({correct / total})")

    return (correct, total)


def get_known_correct_count(checkpoint_filename: str) -> typing.Tuple[int, int]:
    train_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_TRAIN_SUB_DIR,
        download=True,
        transform=TEST_TRANSFORMS
    )
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    validation_dataset = dataset.CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_VALIDATION_SUB_DIR,
        download=True,
        transform=TEST_TRANSFORMS
    )
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    train_correct, train_total = _get_correct_count_of_dataloader(checkpoint_filename, train_dataloader)
    validation_correct, validation_total = _get_correct_count_of_dataloader(checkpoint_filename, validation_dataloader)

    correct = train_correct + validation_correct
    total = train_total + validation_total

    # print(f"Train accuracy: {train_correct}/{train_total} ({train_correct / train_total})")
    # print(f"Validation accuracy: {validation_correct}/{validation_total} ({validation_correct / validation_total})")
    # print(f"Overall accuracy: {correct}/{total} ({correct / total})")

    return (correct, total)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    checkpoint_filenames = [
        "../762-Assignment-2-Checkpoints/0.4-64batch/400.pth",
        "../762-Assignment-2-Checkpoints/0.4-96batch/400.pth",
        "../762-Assignment-2-Checkpoints/0.4-128batch/400.pth",
        "../762-Assignment-2-Checkpoints/0.4-300batch/400.pth",
    ]

    for filename in checkpoint_filenames:
        total_correct, total_total = get_total_correct_count(filename)
        known_correct, known_total = get_known_correct_count(filename)

        remaining_correct = total_correct - known_correct
        remaining_total = total_total - known_total
        print(f"{filename}: remaining accuracy: {remaining_correct}/{remaining_total} ({remaining_correct / remaining_total})")
