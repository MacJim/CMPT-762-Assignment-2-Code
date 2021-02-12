import unittest
from collections import defaultdict

import torch
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

    # MARK: - Length
    @staticmethod
    def get_batch_count(total_count: int, batch_size: int) -> int:
        batch_count = total_count // batch_size
        if ((total_count % batch_size) > 0):
            batch_count += 1
        return batch_count

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

    # MARK: - CUDA tensor
    @unittest.skip("Too time-consuming.")
    def test_train_cuda_tensor(self):
        for batch_size in [1, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_train_data_loader(batch_size=batch_size, shuffle=True, num_workers=6)
                for image, label in dataloader:
                    with self.subTest(image=image, label=label):
                        image_cuda = image.cuda()
                        label_cuda = label.cuda()

    @unittest.skip("Too time-consuming.")
    def test_validation_cuda_tensor(self):
        for batch_size in [1, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_validation_data_loader(batch_size=batch_size, num_workers=6)
                for image, label in dataloader:
                    with self.subTest(image=image, label=label):
                        image_cuda = image.cuda()
                        label_cuda = label.cuda()

    @unittest.skip("Too time-consuming.")
    def test_test_cuda_tensor(self):
        for batch_size in [1, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                dataloader = DatasetTestCase.get_test_data_loader(batch_size=batch_size, num_workers=6)
                for image, label in dataloader:
                    with self.subTest(image=image, label=label):
                        image_cuda = image.cuda()
                        label_cuda = label.cuda()

    # MARK: - Tensor/label shapes
    INPUT_CHANNELS = 3
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 32

    def test_tensor_shapes(self):
        for batch_size in [1, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                dataloaders = [
                    DatasetTestCase.get_train_data_loader(batch_size=batch_size, shuffle=True),
                    DatasetTestCase.get_validation_data_loader(batch_size=batch_size),
                    DatasetTestCase.get_test_data_loader(batch_size=batch_size),
                ]

                for dataloader in dataloaders:
                    with self.subTest(dataloader=dataloader):
                        for image, label in dataloader:
                            # (N, C, H, W)
                            # print(image)
                            self.assertEqual(len(image.shape), 4)
                            self.assertEqual(image.shape[0], batch_size)
                            self.assertEqual(image.shape[1], DatasetTestCase.INPUT_CHANNELS)
                            self.assertEqual(image.shape[2], DatasetTestCase.INPUT_HEIGHT)
                            self.assertEqual(image.shape[3], DatasetTestCase.INPUT_WIDTH)

                            # (N)
                            # So we're not using 1-hot encoding here.
                            # print(label)
                            self.assertEqual(len(label.shape), 1)
                            self.assertEqual(label.shape[0], batch_size)

                            break    # Only test 1 set of tensor.

    # MARK: - Tensor and label values
    def test_tensor_values(self):
        image_max_value = None
        image_min_value = None
        label_max_value = None
        label_min_value = None

        batch_size = 8
        dataloaders = [
            DatasetTestCase.get_train_data_loader(batch_size=batch_size, shuffle=True),
            DatasetTestCase.get_validation_data_loader(batch_size=batch_size),
            DatasetTestCase.get_test_data_loader(batch_size=batch_size),
        ]
        for dataloader in dataloaders:
            with self.subTest(dataloader=dataloader):
                for image, label in dataloader:
                    current_image_max = torch.max(image)
                    current_image_max = current_image_max.item()
                    current_image_min = torch.min(image)
                    current_image_min = current_image_min.item()
                    current_label_max = torch.max(label)
                    current_label_max = current_label_max.item()
                    current_label_min = torch.min(label)
                    current_label_min = current_label_min.item()

                    if (not image_max_value):
                        image_max_value = current_image_max
                    else:
                        image_max_value = max(current_image_max, image_max_value)

                    if (not image_min_value):
                        image_min_value = current_image_min
                    else:
                        image_min_value = min(current_image_min, image_min_value)

                    if (not label_max_value):
                        label_max_value = current_label_max
                    else:
                        label_max_value = max(current_label_max, label_max_value)

                    if (not label_min_value):
                        label_min_value = current_label_min
                    else:
                        label_min_value = min(current_label_min, label_min_value)

        self.assertGreater(image_max_value, 0.99)
        self.assertGreaterEqual(1.0, image_max_value)
        self.assertGreater(0.01, image_min_value)
        self.assertGreaterEqual(image_min_value, 0.0)

        self.assertEqual(label_max_value, 99)
        self.assertEqual(label_min_value, 0)

    def test_test_dataset_label(self):
        """
        The test set only contains 0 labels.

        :return:
        """
        TEST_LEN = 10000

        test_label_occurrences = defaultdict(int)    # All 0 label

        test_dataloader = DatasetTestCase.get_test_data_loader(batch_size=1)
        for _, label in test_dataloader:
            label = torch.reshape(label, (-1,))
            label = label.item()
            test_label_occurrences[label] += 1

        print(test_label_occurrences)
        self.assertEqual(test_label_occurrences, {0: TEST_LEN})


if __name__ == '__main__':
    unittest.main()
