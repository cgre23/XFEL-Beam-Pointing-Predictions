import itertools
import os
import shutil
import unittest
from datetime import timedelta

import numpy as np

from extensions.file_writer import DataStreamRingFileWriter, RingFileWriter


class TestDataStreamRingFileWriter(unittest.TestCase):
    def test_data_integrity(self):
        """
        Tests that the file created by DataStreamRingFileWriter correctly stores the supplied data.
        """

        ring_size = (1, 2, 17, 128)
        sample_shape = ((1,), (2,), (2048,), (1, 1), (1, 2), (2, 1), (2, 2), (4, 2, 1), (1, 2, 4))
        sample_count = (1, 2, 8, 2048)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        sequence_start = (0, 1, 10 ** 16)
        random_seed = (8988, 9090)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for ring_size, sample_shape, sample_count, sample_dtype, sequence_start, random_seed in itertools.product(
                ring_size, sample_shape, sample_count, sample_dtype, sequence_start, random_seed):
            with self.subTest(ring_size=ring_size,
                              sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamRingFileWriter(test_filename + ".ring.npy", ring_size)

                for i in range(sample_count):
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".ring.npy")

                np.testing.assert_array_equal(readback_data,
                                              np.roll(test_data[-ring_size:, :], sample_count % ring_size, axis=0))

        os.remove(test_filename + ".ring.npy")
        os.remove(test_filename + ".ring.meta")

    def test_missing_data_integrity(self):
        """
        Tests that the file created by DataStreamRingFileWriter correctly stores the supplied data, even when there are
        missing samples.
        """

        ring_size = (2000,)
        sample_shape = ((1,), (1, 2), (1, 2, 4))
        sample_count = (4096,)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        missing_sample_count = (1, 128, 1024, 4000)
        sequence_start = (0,)
        random_seed = (8980, 9096)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for ring_size, sample_shape, sample_count, sample_dtype, missing_sample_count, sequence_start, random_seed in \
                itertools.product(ring_size, sample_shape, sample_count, sample_dtype, missing_sample_count,
                                  sequence_start, random_seed):
            with self.subTest(ring_size=ring_size,
                              sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              missing_sample_count=missing_sample_count,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamRingFileWriter(test_filename + ".ring.npy", ring_size)

                indices = set(range(sample_count))
                missing_indices = set(rg.choice(
                    np.arange(1, sample_count), missing_sample_count, False))  # first sample cannot be missing
                indices.difference(missing_indices)

                test_data[tuple(indices), :] = data_writer._nan_equivalent(test_data.dtype)

                for i in indices:
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".ring.npy")

                np.testing.assert_array_equal(readback_data,
                                              np.roll(test_data[-ring_size:, :], sample_count % ring_size, axis=0))

        os.remove(test_filename + ".ring.npy")
        os.remove(test_filename + ".ring.meta")

    def test_shuffled_data_integrity(self):
        """
        Tests that the file created by DataStreamRingFileWriter correctly stores the supplied data, even when there are
        unordered samples.
        """

        ring_size = (4, 100)
        sample_shape = ((1,), (1, 2), (1, 2, 4))
        sample_count = (8, 256, 2048)
        sample_dtype = (np.float, np.ubyte, np.complex, np.longlong, np.clongfloat)
        sequence_start = (0,)
        random_seed = (8982, 9091)

        test_filename = "test" + str(np.random.default_rng().integers(1e3, 1e4))

        for ring_size, sample_shape, sample_count, sample_dtype, sequence_start, random_seed in itertools.product(
                ring_size, sample_shape, sample_count, sample_dtype, sequence_start, random_seed):
            with self.subTest(ring_size=ring_size,
                              sample_shape=sample_shape,
                              sample_count=sample_count,
                              sample_dtype=sample_dtype,
                              sequence_start=sequence_start,
                              random_seed=random_seed):
                rg = np.random.Generator(np.random.PCG64(random_seed))
                test_data = rg.uniform(0, 2 ** 32 - 1, (sample_count, *sample_shape)).astype(sample_dtype)

                data_writer = DataStreamRingFileWriter(test_filename + ".ring.npy", ring_size)

                indices = list(range(1, sample_count))
                rg.shuffle(indices)
                indices.insert(0, 0)  # first sample always has to come first
                for i in indices:
                    data_writer.store_sample(test_data[i, :], sequence_start + i)
                data_writer.close()

                readback_data = np.load(test_filename + ".ring.npy")

                np.testing.assert_array_equal(readback_data,
                                              np.roll(test_data[-ring_size:, :], sample_count % ring_size, axis=0))

        os.remove(test_filename + ".ring.npy")
        os.remove(test_filename + ".ring.meta")

    def test_start_sample(self):
        """Tests that the start_sample field in the metadata file points to the correct position (ring seam)."""


class TestRingFileWriter(unittest.TestCase):
    def test_split_delay(self):
        """Tests that the correct number of samples is stored after a split."""

        test_foldername = "test" + str(np.random.default_rng().integers(1e3, 1e4))
        channel_name = 'test_chan'
        ring_file_time = timedelta(seconds=10)
        split_delay = 10
        sequence_start = 0

        data_writer = RingFileWriter({channel_name}, set(), ring_file_time, split_signal_delay=split_delay,
                                     ring_dir_name=test_foldername, memory_buffering=True)
        for i in range(140):
            data_writer.process(channel_name, i, sequence_start + i)
        data_writer.signal(None)
        for i in range(140, 200):
            data_writer.process(channel_name, i, sequence_start + i)
        data_writer.close()

        readback_data = np.load(os.path.join(test_foldername + '-000', channel_name + '.npy'))

        np.testing.assert_array_equal(readback_data,
                                      np.expand_dims(np.roll(np.arange(50, 150), 50), 1))

        shutil.rmtree(test_foldername + '-000')
        shutil.rmtree(test_foldername + '-001')


if __name__ == '__main__':
    unittest.main()
