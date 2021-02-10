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

    # MARK: - Get data loaders
    @staticmethod
    def get_train_data_loader(batch_size: int, shuffle: bool, num_workers=0) -> data.DataLoader:
        dataset = DatasetTestCase.get_train_dataset()
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def get_validation_data_loader(batch_size: int, num_workers=0) -> data.DataLoader:
        dataset = DatasetTestCase.get_validation_dataset()
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    @staticmethod
    def get_test_data_loader(batch_size: int, num_workers=0) -> data.DataLoader:
        dataset = DatasetTestCase.get_test_dataset()
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # MARK: - Get batch count
    @staticmethod
    def get_batch_count(total_count: int, batch_size: int) -> int:
        batch_count = total_count // batch_size
        if ((total_count % batch_size) > 0):
            batch_count += 1
        return batch_count

    # MARK: - Length
    def test_train_length(self):
        TRAIN_LEN = 45000

        dataset = DatasetTestCase.get_train_dataset()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), TRAIN_LEN)

        dataloader = DatasetTestCase.get_train_data_loader(batch_size=1, shuffle=False)
        self.assertGreater(len(dataloader), 0)
        self.assertEqual(len(dataloader), TRAIN_LEN)

        for batch_size in range(2, 21):
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_train_data_loader(batch_size=batch_size, shuffle=False)
                self.assertGreater(len(dataloader), 0)
                self.assertEqual(len(dataloader), DatasetTestCase.get_batch_count(TRAIN_LEN, batch_size))

    def test_validation_length(self):
        VALIDATION_LEN = 5000

        dataset = DatasetTestCase.get_validation_dataset()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), VALIDATION_LEN)

        dataloader = DatasetTestCase.get_validation_data_loader(batch_size=1)
        self.assertGreater(len(dataloader), 0)
        self.assertEqual(len(dataloader), VALIDATION_LEN)

        for batch_size in range(2, 21):
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_validation_data_loader(batch_size=batch_size)
                self.assertGreater(len(dataloader), 0)
                self.assertEqual(len(dataloader), DatasetTestCase.get_batch_count(VALIDATION_LEN, batch_size))

    def test_test_length(self):
        TEST_LEN = 10000

        dataset = DatasetTestCase.get_test_dataset()
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), TEST_LEN)

        dataloader = DatasetTestCase.get_test_data_loader(batch_size=1)
        self.assertGreater(len(dataloader), 0)
        self.assertEqual(len(dataloader), TEST_LEN)

        for batch_size in range(2, 21):
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_test_data_loader(batch_size=batch_size)
                self.assertGreater(len(dataloader), 0)
                self.assertEqual(len(dataloader), DatasetTestCase.get_batch_count(TEST_LEN, batch_size))


if __name__ == '__main__':
    unittest.main()
