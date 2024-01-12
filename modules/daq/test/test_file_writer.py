import itertools
import os
import unittest

import numpy as np

from extensions.file_writer import DataStreamFileWriter


class TestDataStreamFileWriter(unittest.TestCase):
    def test_data_integrity(self):
        """
        Tests that the file created by DataStreamFileWriter correctly stores the supplied data.
        """

        sample_shape = ((1,), (2,), (2048,), (1, 1), (1, 2), (2, 1), (2, 2), (4, 2, 1), (1, 2, 4))
        sample_count = (1, 2, 8, 2048)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        sequence_start = (0, 1, 10 ** 16)
        random_seed = (8988, 9090)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for sample_shape, sample_count, sample_dtype, sequence_start, random_seed in itertools.product(
                sample_shape, sample_count, sample_dtype, sequence_start, random_seed):
            with self.subTest(sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamFileWriter(test_filename + ".npy")

                for i in range(sample_count):
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".npy")

                np.testing.assert_array_equal(readback_data, test_data)

        os.remove(test_filename + ".npy")
        os.remove(test_filename + ".meta")

    def test_missing_data_integrity(self):
        """
        Tests that the file created by DataStreamFileWriter correctly stores the supplied data, even when there are
        missing samples.
        """

        sample_shape = ((1,), (1, 2), (1, 2, 4))
        sample_count = (2048,)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        missing_sample_count = (1, 128, 1024, 2000)
        sequence_start = (0,)
        random_seed = (8980, 9096)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for sample_shape, sample_count, sample_dtype, missing_sample_count, sequence_start, random_seed in \
                itertools.product(sample_shape, sample_count, sample_dtype, missing_sample_count, sequence_start,
                                  random_seed):
            with self.subTest(sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              missing_sample_count=missing_sample_count,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamFileWriter(test_filename + ".npy")

                indices = set(range(sample_count))
                missing_indices = set(rg.choice(
                    np.arange(1, sample_count), missing_sample_count, False))  # first sample cannot be missing
                indices.difference(missing_indices)

                test_data[tuple(indices), :] = data_writer._nan_equivalent(test_data.dtype)

                for i in indices:
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".npy")

                np.testing.assert_array_equal(readback_data, test_data)

        os.remove(test_filename + ".npy")
        os.remove(test_filename + ".meta")

    def test_shuffled_data_integrity(self):
        """
        Tests that the file created by DataStreamFileWriter correctly stores the supplied data, even when there are
        unordered samples.
        """

        sample_shape = ((1,), (1, 2), (1, 2, 4))
        sample_count = (1, 2, 8, 2048)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        sequence_start = (0,)
        random_seed = (8982, 9091)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for sample_shape, sample_count, sample_dtype, sequence_start, random_seed in itertools.product(
                sample_shape, sample_count, sample_dtype, sequence_start, random_seed):
            with self.subTest(sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamFileWriter(test_filename + ".npy")

                indices = list(range(1, sample_count))
                rg.shuffle(indices)
                indices.insert(0, 0)  # first sample always has to come first
                for i in indices:
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".npy")

                np.testing.assert_array_equal(readback_data, test_data)

        os.remove(test_filename + ".npy")
        os.remove(test_filename + ".meta")


if __name__ == '__main__':
    unittest.main()
