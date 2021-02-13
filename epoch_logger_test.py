import unittest
import os

from epoch_logger import log_epoch_details_to_file, read_epoch_details_from_file


class EpochLoggerTestCase (unittest.TestCase):
    def test_write_then_read(self):
        tmp_filename = "/tmp/epoch_logger_test_write_then_read.csv"
        if (os.path.isfile(tmp_filename)):
            os.unlink(tmp_filename)

        original_contents = [
            [1, 10, 4, 0.4, 0.444, 0.0444, 30.5],
            [2, 10, 5, 0.5, 0.555, 0.0555, 32.5],
            [2, 10, 6, 0.6, 0.666, 0.0666, 34.5],
        ]

        for i, c in enumerate(original_contents):
            with self.subTest(i=i):
                log_epoch_details_to_file(c[0], c[1], c[2], c[3], c[4], c[5], c[6], tmp_filename)
                read_contents = read_epoch_details_from_file(tmp_filename)
                self.assertEqual(original_contents[: (i + 1)], read_contents)


if __name__ == '__main__':
    unittest.main()
