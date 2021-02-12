import typing
import csv
import os
import argparse

import torch
from torch.utils import data
from torchvision import transforms

import constant
from dataset import CIFAR100_SFU_CV
from densenet import DenseNet762


# MARK: - Constants
TEST_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
TEST_BATCH_SIZE: typing.Final = 1000


def main(checkpoint_filename: str, csv_filename: str):
    test_dataset = CIFAR100_SFU_CV(
        root=constant.DATASET_ROOT_DIR,
        fold=constant.DATASET_TEST_SUB_DIR,
        download=True,
        transform=TEST_TRANSFORMS
    )
    test_dataloader = data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    network = DenseNet762()
    network = network.cuda()
    network.load_state_dict(torch.load(checkpoint_filename))
    network.eval()

    total = 0
    predictions = []

    # Infer.
    with torch.no_grad():
        for image, label in test_dataloader:
            image = image.cuda()
            label = label.cuda()

            prediction = network(image)
            _, max_prediction_indices = torch.max(prediction, -1)
            predictions.extend(list(max_prediction_indices.cpu().numpy()))
            total += label.shape[0]

    # Save to CSV.
    with open(csv_filename, "w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id", "Prediction1"])
        for l_i, label in enumerate(predictions):
            wr.writerow([str(l_i), str(label)])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_filename", type=str, default="../762-Assignment-2-Checkpoints/0.3/200.pth")
    parser.add_argument("--csv_filename", type=str, default="predictions.csv")
    args = parser.parse_args()

    main(args.checkpoint_filename, args.csv_filename)
