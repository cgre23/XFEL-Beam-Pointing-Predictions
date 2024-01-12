import io
import json
import logging
import os

import numpy as np
import numpy.lib.format as npformat

from dxmaf.util.buffered_ring_writer import BufferedRingWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())


class DuplicateDataSampleError(ValueError):
    """Data sample has sequence number that has already been received before."""


class InconsistentDataSampleError(ValueError):
    """Data sample is not consistent with specification passed to __init__ or derived from first data sample."""


class DataStreamFileWriter:
    """
    A writer object for storing data samples from a continuous stream of consistent numpy data samples to a numpy file.

    Supports data ordering for non-sequential arrival of samples.

    Missing data is represented as the channel's data type NaN value.

    TODO: Implement parameter for load balancing flushing (i.e. a macropulse offset at which the buffer is flushed).
    """

    @property
    def sequence_id(self):
        """Next expected sequence id (i.e. last received sequence id + 1)."""
        return self._cur_seq_id

    @property
    def missing_sequence_ids(self):
        """
        List of sequence ids before the current sequence id and after the first received sequence id, for which no data
        has yet been received.
        """
        return tuple(self._missing_seq_ids)

    @property
    def size(self):
        """Size of the current data file being written, in bytes."""
        return 0 if self._file is None else self._file.seek(0, 2)

    @property
    def dtype(self):
        """Numpy data type of the data stream."""
        return self._dtype

    @property
    def filename(self):
        return self._filename + self._fileext + (f".{self._filenumber}" if self._filenumber > 0 else "")

    @property
    def metadata_filename(self):
        return self._filename + ".meta" + (f".{self._filenumber}" if self._filenumber > 0 else "")

    @property
    def _sample_byte_size(self):
        """Size  in bytes of a data sample."""
        return self._dtype.itemsize * int(np.prod(self._sample_shape))

    def __init__(self, filename, dtype=None, sample_shape=None, store_meta_info=True, static_meta_info=None):
        """
        Initializes a new writer object.

        :param filename:         File path and name with extension to which data is written.
        :param dtype:            Predefined expected data type of data samples. Optional.
        :param sample_shape:     Predefined expected shape of expected data samples. Optional.
        :param store_meta_info:  Whether to create a metadata file.
        :param static_meta_info: Static meta information to be stored in the metadata file.
        """
        self._dtype = dtype
        self._sample_shape = sample_shape
        self._header_size = None
        self._file = None
        self._filename, self._fileext = os.path.splitext(filename)
        self._filenumber = 0
        self._store_meta_info = store_meta_info
        self._first_seq_id = None
        self._cur_seq_id = None  # next expected sequence id
        self._missing_seq_ids = set()

        self.static_meta_info = static_meta_info if static_meta_info is not None else {}

    @staticmethod
    def _nan_equivalent(dtype):
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.inexact):
            return np.NaN
        else:
            raise NotImplementedError("no NaN equivalent for non-numeric data types.")

    def _auto_init(self, data, sequence_id):
        """
        Automatically initializes class members based on information contained in first stored data sample.

        :param data:        First data sample.
        :param sequence_id: First data sample sequence id.
        """
        self._first_seq_id = sequence_id
        self._cur_seq_id = sequence_id

        if self._dtype is None:
            self._dtype = data.dtype
        if self._sample_shape is None:
            self._sample_shape = data.shape

        self._create_datafile()

    def _create_datafile(self):
        self._file = open(self.filename, 'wb')
        self._write_provisional_header()

    def _close_file(self):
        """
        Writes an updated numpy header to the beginning of the file and closes it.
        """
        if self._file is None:
            return

        header = npformat.header_data_from_array_1_0(np.empty((1,), self._dtype))
        if self._cur_seq_id is not None and self._first_seq_id is not None:
            header['shape'] = ((self._cur_seq_id - self._first_seq_id),) + self._sample_shape
        else:
            header['shape'] = (0,)

        self._file.seek(0)
        npformat.write_array_header_1_0(self._file, header)

        new_header_size = self._file.tell()
        if self._header_size and new_header_size != self._header_size:
            raise RuntimeError(f"Updated header size ({new_header_size} bytes) does not match preallocated header size "
                               f"({self._header_size} bytes)")

        self._file.close()
        self._file = None

    def _seek_sequence_id(self, sequence_id):
        """
        Seeks the position in the data file, where the data sample with the given sequence id is located.

        :param sequence_id: Sequence id of the data sample to be located in the file.
        """
        if sequence_id < self._first_seq_id:
            raise IndexError(f"Sequence id '{sequence_id}' lies before first sequence id in file "
                             f"'{self._first_seq_id}'")

        distance = self._cur_seq_id - sequence_id

        self._file.seek(-distance * self._sample_byte_size, 2)

    def _get_meta_info(self):
        """
        Constructs a dictionary with meta information about the current data stream. Does not include meta data
        contained in header of npy storage file.

        :return: Metainformation
        """
        meta_info = {
            'file_sequence_number': self._filenumber,
            'first_sequence_id':    self._first_seq_id,
            'missing_sequence_ids': tuple(self._missing_seq_ids)
        }

        meta_info.update(self.static_meta_info)

        return meta_info

    def _write_meta_info_file(self):
        """
        Writes meta information to a .meta file with the same base filename as the storage file.
        """
        with open(self.metadata_filename, 'w+') as f:
            json.dump(self._get_meta_info(), f, indent=4)
            f.write('\n')

    def _write_provisional_header(self):
        """
        Writes a provisional header to the storage file  as to allocate space for the final header, which is to be
        written with the latest information before closing the file.
        """

        if self._sample_shape is None or self._dtype is None:
            raise RuntimeError("Cannot write provisional header without data type and sample shape information!")

        self._file.seek(0)
        npformat.write_array_header_1_0(self._file,
                                        npformat.header_data_from_array_1_0(np.empty((1,) + self._sample_shape,
                                                                                     self._dtype)))
        self._header_size = self._file.tell()

    def _replace_sequence_id(self, data, sequence_id):
        """
        Replaces data sample with specified sequence id in the storage file with new data.

        :param data:        New data to be put at the position given by `sequence_id`.
        :param sequence_id: Sequence id of the data sample to be replaced with `data`.
        """

        self._seek_sequence_id(sequence_id)
        self._file.write(data.tobytes())

    def store_sample(self, data, sequence_id):
        """
        Stores a data sample to the current file.

        The first stored sample will determine the expected shape of future data. Similarly, the first sample id
        determines the start of the data sequence and needs not start at any particular number.

        Subsequent duplicate data samples (with identical id) and inconsistently sized data samples will be ignored.

        :param data:        Consistently size data sample, that can be interpreted as a 1d numpy array.
        :param sequence_id: Sequence number for the data sample.
        """
        data = np.atleast_1d(data)

        if self._first_seq_id is None:
            self._auto_init(data, sequence_id)

        if data.dtype != self._dtype:
            raise InconsistentDataSampleError(
                f"Data sample type inconsistent, expected: {self._dtype}, got: {data.dtype}")
        elif data.shape != self._sample_shape:
            raise ValueError(f"Data sample shape inconsistent, expected: {self._sample_shape}, got: {data.shape}")
        elif sequence_id < self._first_seq_id:
            logger.debug(f"Received data sample whose sequence id '{sequence_id}' is smaller than "
                         f"initial sample sequence id '{self._first_seq_id}'. Ignoring...")
            return

        self._seek_sequence_id(self._cur_seq_id)

        distance = sequence_id - self._cur_seq_id
        if distance < 0:
            if sequence_id in self._missing_seq_ids:
                logger.info(f"Received late sample for id '{sequence_id}'.")
                self._replace_sequence_id(data, sequence_id)
                self._missing_seq_ids.remove(sequence_id)
            else:
                raise DuplicateDataSampleError(f"Duplicate data sample for sequence id '{sequence_id}'")
        elif distance > 0:
            logger.info(f"Missed data for sequence id '{self._cur_seq_id} -> {sequence_id - 1}'. "
                        "Putting NaN equivalent...")
            self._missing_seq_ids.update(range(self._cur_seq_id, sequence_id))
            self._file.write(
                np.full((distance, *self._sample_shape), self._nan_equivalent(self._dtype), self._dtype).tobytes())
            self._file.write(data.tobytes())
            self._cur_seq_id += distance + 1

        else:  # data_distance == 0
            self._file.write(data.tobytes())
            self._cur_seq_id += 1

    def split(self, filename=...):
        """
        Flushes buffers, writes meta information file, closes the current storage file and
        continues data recording in a new storage file.

        :param filename: Name of the new file being created, with extension. If not supplied, uses previous filename
                         with an appended sequence number before the extension.
        """

        if self._first_seq_id is None:  # we haven't even started yet
            if filename is not ...:
                self._filename, self._fileext = os.path.splitext(filename)
                self._filenumber = 0
            return

        self._write_meta_info_file()
        self._close_file()

        self._filenumber += 1
        self._missing_seq_ids = set()
        self._first_seq_id = self._cur_seq_id

        if filename is not ...:
            self._filename, self._fileext = os.path.splitext(filename)
            self._filenumber = 0

        self._create_datafile()

    def close(self):
        """
        Stops data recording, flushes all buffers, writes meta information file and closes the storage file.

        The object must not be used after closing it.
        """

        if self._file is not None:
            self._write_meta_info_file()
            self._close_file()


class DataStreamRingFileWriter(DataStreamFileWriter):
    """
    A writer object for storing data samples from a continuous stream of consistent numpy data samples to a
    ring-buffered numpy file.

    Supports data ordering for non-sequential arrival of samples.

    Missing data is represented as the channel's data type NaN value.
    """

    def __init__(self, filename, ringfile_size, dtype=None, sample_shape=None, store_meta_info=True,
                 static_meta_info=None, memory_buffering=False):
        """
        Initializes the DataSteamRingFileWriter object.

        :param filename:         File path and name with extension to which data is written.
        :param ringfile_size:    Size of the ring file in samples.
        :param dtype:            Predefined expected data type of data samples. Optional.
        :param sample_shape:     Predefined expected shape of expected data samples. Optional.
        :param store_meta_info:  Whether to create a metadata file.
        :param static_meta_info: Static meta information to be stored in the metadata file.
        :param memory_buffering: Keep ring in memory until it is closed, then write to file system.
        """
        super().__init__(filename, dtype, sample_shape, store_meta_info, static_meta_info)
        self._ringfile_size = ringfile_size
        self._memory_buffering = memory_buffering

    def _close_file(self):
        if self._memory_buffering:
            diskfile = open(self.filename, 'wb')
            diskfile.write(self._file.raw.getbuffer())
            self._file.close()
            self._file = diskfile

        super()._close_file()

    def _create_datafile(self):
        if self._memory_buffering:
            self._file = io.BytesIO(np.empty(128 + self._ringfile_size * self._sample_byte_size, dtype=np.byte))
        else:
            self._file = open(self.filename, 'wb')
        self._write_provisional_header()
        self._file = BufferedRingWriter(self._file, self._ringfile_size * self._sample_byte_size,
                                        header_size=self._header_size)

    def _get_meta_info(self):
        """
        Constructs a dictionary with meta information about the current data stream. Does not include meta data
        contained in header of npy storage file.

        :return: Metainformation
        """
        meta_info = super()._get_meta_info()
        meta_info.update({
            'start_sample': self._file.seam_position // self._sample_byte_size
        })

        return meta_info

    def store_sample(self, data, sequence_id):
        """
        Stores a data sample to the current ring file.

        The first stored sample will determine the expected shape of future data. Similarly, the first sample id
        determines the start of the data sequence and needs not start at any particular number.

        Subsequent duplicate data samples (with identical id) and inconsistently sized data samples will be ignored.

        :param data:        Consistently size data sample, that can be interpreted as a 1d numpy array.
        :param sequence_id: Sequence number for the data sample.
        """
        super().store_sample(data, sequence_id)

        if self._cur_seq_id > self._first_seq_id + self._ringfile_size:
            self._missing_seq_ids -= set(range(self._first_seq_id, self._cur_seq_id - self._ringfile_size))
            self._first_seq_id = self._cur_seq_id - self._ringfile_size
