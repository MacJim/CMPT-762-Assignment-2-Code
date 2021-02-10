import unittest

import torch

from model import DenseNet


class ModelTestCase (unittest.TestCase):
    # MARK: - Test CUDA
    @unittest.skip
    def test_cuda(self):
        n_classes = 100

        network = DenseNet(n_classes)
        network = network.cuda()
        # print(network)

    # MARK: - Test calculation using dummy inputs
    INPUT_CHANNELS = 3
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 32

    def test_dummy_value(self):
        n_classes = 100
        network = DenseNet(n_classes)
        network = network.cuda()

        network.train()
        # Batch size 1 is unavailable because we have batch norm layers.
        for batch_size in [2, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                image = torch.rand(
                    (batch_size, ModelTestCase.INPUT_CHANNELS, ModelTestCase.INPUT_HEIGHT, ModelTestCase.INPUT_WIDTH))
                image = image.cuda()
                output = network(image)

                # Test shape.
                # print(output.shape)
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(output.shape[1], n_classes)

                # Test sum.
                # DO NOT include Softmax in the model because `nn.CrossEntropyLoss` includes that.
                output_sum = torch.sum(output, dim=1)
                for current_sum in output_sum.tolist():
                    self.assertNotAlmostEqual(current_sum, 1.)

        network.eval()
        for batch_size in [1, 2, 4, 8, 16]:
            with self.subTest(batch_size=batch_size):
                image = torch.rand((batch_size, ModelTestCase.INPUT_CHANNELS, ModelTestCase.INPUT_HEIGHT, ModelTestCase.INPUT_WIDTH))
                image = image.cuda()
                output = network(image)

                # Test shape.
                # print(output.shape)
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(output.shape[1], n_classes)

                # Test sum.
                output_sum = torch.sum(output, dim=1)
                for current_sum in output_sum.tolist():
                    self.assertNotAlmostEqual(current_sum, 1.)


if __name__ == '__main__':
    unittest.main()
