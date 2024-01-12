from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Mapping, Set

import numpy

TaggedData = namedtuple('TaggedData', ['data', 'sequence_id', 'timestamp'])

class DataSubscriber(metaclass=ABCMeta):
    """Abstract base class for plugins subscribing to data from DOOCS."""

    def __init__(self, channels: Set[str]):
        """
        Initializes the DataSubscriber object.

        :param channels: Set (unique sequence) of DOOCS channel addresses for which `process` will be called in the
                         event of new data.
        """
        self._subscribed_channels = channels

    @abstractmethod
    def process(self, channel: str, data, sequence_id: int, timestamp: float) -> None:
        """
        Process data from a channel previously subscribed to.

        For performance, data is passed as reference and is read-only and deep-copies should be avoided.

        :param channel:     DOOCS address of the channel from which `data` was received.
        :param data:        Read-only data sample from the previously subscribed to channel specified in `channel`.
        :param sequence_id: Sequence ID (macropulse number) of the data sample.
        :param timestamp:   Timestamp of the data sample.
        """
        pass

    def close(self):
        """
        Called when the subscriber is no longer needed. Should be implemented to perform clean-up actions, such as
        closing files etc.
        """
        pass


class BufferedDataSubscriber(DataSubscriber):
    """Abstract base class for plugins subscribing to data from DOOCS. Uses buffering to guarantee completeness."""

    def __init__(self, channels: Set[str], buffer_size: int = 8):
        super().__init__(channels)

        self.channels = channels
        self.max_buffer_size = buffer_size
        self.buffer = {}

    @abstractmethod
    def process_incomplete(self, dataset: Mapping[str, numpy.ndarray], sequence_id: int) -> None:
        """
        Process an incomplete dataset from channels previously subscribed to, before the data is discarded from the
        buffer.

        :param dataset:     Mapping of DOOCS property addresses to a corresponding data sample with synchronous
                            sequence id (macropulse number). Completeness is guaranteed.
        :param sequence_id: Sequence ID (macropulse number) of all data samples in `dataset`.
        :return:            None
        """
        pass

    @abstractmethod
    def process_complete(self, dataset: Mapping[str, TaggedData], sequence_id: int) -> None:
        """
        Process a complete dataset from channels previously subscribed to.

        :param dataset:     Mapping of DOOCS property addresses to a corresponding data sample with synchronous
                            sequence id (macropulse number). Completeness is guaranteed.
        :param sequence_id: Sequence ID (macropulse number) of all data samples in `dataset`.
        :return:            None
        """
        print(sequence_id, dataset)
       

    def process(self, channel: str, data, sequence_id: int, timestamp: float) -> None:
        """
        Process data from a channel previously subscribed to.

        Buffers data received from individual channels until all data for any given sequence number has been received
        from all channels and then calls `process_complete` to process the complete data set. Data older than the
        specified maximum buffer size is discarded.

        :param channel:     DOOCS address of the channel from which `data` was received.
        :param data:        Read-only data sample from the previously subscribed to channel specified in `channel`.
        :param sequence_id: Sequence ID (macropulse number) of the data sample.
        :param timestamp:   Timestamp of the data sample.
        :return:            None
        """
        highest_sequence_id = max([*self.buffer.keys(), sequence_id])
        if sequence_id < (highest_sequence_id - self.max_buffer_size):
            return

        for key in self.buffer.keys():
            if key < (highest_sequence_id - self.max_buffer_size):
                self.process_incomplete(self.buffer[key], key)
                del self.buffer[key]

        self.buffer.setdefault(sequence_id, {})[channel] = TaggedData(data, sequence_id, timestamp)

        if len(self.buffer[sequence_id]) == len(self.channels):
            self.process_complete(self.buffer[sequence_id], sequence_id)
            del self.buffer[sequence_id]
