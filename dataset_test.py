import unittest

from torch.utils import data
from torchvision import transforms

from dataset import CIFAR100_SFU_CV


class DatasetTestCase (unittest.TestCase):
    # MARK: - Constants
    DATASET_ROOT_DIR = "data/"
    DATASET_TRAIN_SUB_DIR = "train"
    DATASET_VALIDATION_SUB_DIR = "val"
    DATASET_TEST_SUB_DIR = "test"

    TRAIN_TRANSFORM = transforms.Compose([transforms.ToTensor()])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    # MARK: - Get datasets
    @staticmethod
    def get_train_dataset() -> data.Dataset:
        return CIFAR100_SFU_CV(
            root=DatasetTestCase.DATASET_ROOT_DIR,
            fold=DatasetTestCase.DATASET_TRAIN_SUB_DIR,
            download=True,
            transform=DatasetTestCase.TRAIN_TRANSFORM
        )

    @staticmethod
    def get_validation_dataset() -> data.Dataset:
        return CIFAR100_SFU_CV(
            root=DatasetTestCase.DATASET_ROOT_DIR,
            fold=DatasetTestCase.DATASET_VALIDATION_SUB_DIR,
            download=True,
            transform=DatasetTestCase.TEST_TRANSFORM
        )

    @staticmethod
    def get_test_dataset() -> data.Dataset:
        return CIFAR100_SFU_CV(
            root=DatasetTestCase.DATASET_ROOT_DIR,
            fold=DatasetTestCase.DATASET_TEST_SUB_DIR,
            download=True,
            transform=DatasetTestCase.TEST_TRANSFORM
        )

    # MARK: - Length
    def test_train_length(self):
        dataset = DatasetTestCase.get_train_dataset()

        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), 45000)

    def test_validation_length(self):
        dataset = DatasetTestCase.get_validation_dataset()

        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), 5000)

    def test_test_length(self):
        dataset = DatasetTestCase.get_test_dataset()

        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), 10000)


if __name__ == '__main__':
    unittest.main()
