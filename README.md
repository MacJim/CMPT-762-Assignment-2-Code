# CIFAR-100 Classification with DenseNet

- [CIFAR-100 Classification with DenseNet](#cifar-100-classification-with-densenet)
    - [Prerequisites](#prerequisites)
    - [Directory Structure](#directory-structure)
    - [Train](#train)
        - [Visualize Training Loss and Validation Accuracy](#visualize-training-loss-and-validation-accuracy)
    - [Infer](#infer)
        - [Pre-trained Network Weights](#pre-trained-network-weights)
    - [References](#references)

Project requirements: <https://docs.google.com/document/d/1ZUdQ7c2X7y_KWobwZG0y0uFpwe__K1Hs1I_rv5jR3ws/edit?usp=sharing>

## Prerequisites

- Python 3.8 or later
    - You may remove `typing.Final` type hints to use this project with earlier Python versions
- Install all requirements in `requirements.txt`
    - Using `pip`: `pip install -r requirements.txt`

## Directory Structure

- `constant.py`: Constants
- `dataset.py`: The CIFAR_SFU dataset, copied from the handout notebook, unmodified
- `densenet.py`: The DenseNet model
- `epoch_logger.py`: Logs the train/validation losses/accuracies into 2 files: `train_log.csv` and `validation_log.csv`
- `epoch_visualizer.py`: Generates the training loss & validation plot
    - Copied from the handout notebook
    - Slightly modified to show fewer x labels (showing 400 x labels for 400 epochs is unfeasible)
- `infer.py`: Generates the Kaggle submission csv using the test set
- `train.py`: The training script
- Files with filenames ending with `_test` are Python unit tests

Most scripts are using the `argparse` module to parse their command line arguments.
Use `-h` on a script to view all available arguments.

## Train

Run the `train.py` script with arguments specifying where to save the checkpoints and logs.

```bash
python train.py --checkpoint_save_dir=checkpoints --train_log_filename=checkpoints/train_log.csv --validation_log_filename=checkpoints/validation_log.csv
```

- Network weights are saved in `checkpoint_save_dir`
- Training and validation logs (accuracies, losses, etc.) are saved in `train_log_filename` and `validation_log_filename` respectively

Training takes about 4 hours on a machine with i7-9700K and RTX 2080.

### Visualize Training Loss and Validation Accuracy

The `epoch_visualizer.py` script reads `checkpoints/train_log.csv` and `checkpoints/validation_log.csv` to generate a plot of training losses and validation accuracies.
It then saves it as `checkpoints/plot.png`.

## Infer

Run the `infer.py` script with arguments specifying the checkpoint location and prediction csv location.

```bash
python infer.py --checkpoint_filename=../0.4-128batch/400.pth --csv_filename=predictions.csv
```

- Loads network weights from `checkpoint_filename`
- Saves test set predictions in `csv_filename`

### Pre-trained Network Weights

Pre-trained network weights are available at: <https://github.com/MacJim/CMPT-762-Assignment-2-Checkpoints>

The `0.4-128batch/400.pth` weights have the best test accuracy 0.76500 (runner-up in [our Kaggle competition](https://www.kaggle.com/c/sfu-cmpt-image-classification-2021-spring/leaderboard)).

## References

- SFU CMPT 762 home page: <https://www2.cs.sfu.ca/~furukawa/cmpt762-2021-spring/>
- CIFAR-100 dataset: <https://www.cs.toronto.edu/~kriz/cifar.html>
- DenseNet: <https://arxiv.org/abs/1608.06993>
