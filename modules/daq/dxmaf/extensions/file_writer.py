import json
import logging
import os
import tempfile
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Set, Any, Optional, Mapping

from dxmaf.data_subscriber import DataSubscriber
from dxmaf.event_publisher import EventPublisher
from dxmaf.event_subscriber import EventSubscriber
from dxmaf.util.numpy_file_writer import DataStreamFileWriter, DataStreamRingFileWriter, DuplicateDataSampleError, \
    InconsistentDataSampleError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileWriter(DataSubscriber, EventSubscriber, EventPublisher):
    """
    DxMAF module that writes data from subscribed channels to numpy and metadata files.

    # TODO: Add file system free space check.
    """

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def record_timestamps(self):
        return self._record_timestamps

    def _init_file_writers(self) -> None:
        """
        Initializes a file writer for each channel.
        """

        for address in self._subscribed_channels:
            filepath = self.output_dir / (address.replace('/', '-') + '.npy')
            self.channels[address] = DataStreamFileWriter(filepath,
                                                          static_meta_info={'address': address})

            if self.record_timestamps:
                self.timestamps[address] = DataStreamFileWriter(filepath.with_suffix('.ts.npy'),
                                                                static_meta_info={'address': address,
                                                                                  'type': 'timestamp'})

    def _split(self) -> None:
        """
        Closes the current set of files and moves to the next (for example so that the previous data can be moved
        elsewhere). Publishes an event with a list of all closed filenames.
        """
        filepaths = []
        for channel in self.channels.values():
            filepaths.append(channel.filename)
            filepaths.append(channel.metadata_filename)
            channel.split()  # This changes the filename so has to come last.

        if self.record_timestamps:
            for timestamp in self.timestamps.values():
                filepaths.append(timestamp.filename)
                timestamp.split()

        # ensures files exist (they might not if no data was received)
        filepaths = [fp for fp in filepaths if os.path.isfile(fp)]

        self.publish(filepaths)

    def __init__(self, channels: Set[str], output_dir: PathLike = "run_%Y-%m-%d_%H%M%S",
                 record_timestamps: Optional[bool] = False, split_size: Optional[int] = None,
                 split_signal_delay: Optional[int] = 1, static_metadata_files: Optional[Mapping[str, str]] = None,
                 topics: Optional[Set[str]] = None):
        """
        Initializes the FileWriter object.

        :param channels:              Set (unique sequence) of DOOCS channel addresses for which `process` will be
                                      called in the event of new data.
        :param output_dir:            Output directory path where the data files are stored. Allows for datestring
                                      tokens.
        :param record_timestamps:     Record a timestamp for each sequence id.
        :param split_size:            Data file size (per channel) after which the current file is closed and writing
                                      continues in the next file. None indicates no limit.
        :param split_signal_delay:    Delay in sequence id's after which a split signal is executed.
        :param static_metadata_files: Mapping of DOOCS channel addresses (same as `channels`) to files containing
                                      static metdata information.
        :param topics:                Topics to which to publish in the event of a split.
        """
        self.split_size = split_size
        self.split_signal_delay = split_signal_delay

        self._output_dir = Path(datetime.now().strftime(str(output_dir)))
        self._record_timestamps = record_timestamps
        self._cur_seq_id = 0
        self._split_at_seq_id = None

        DataSubscriber.__init__(self, channels)
        EventPublisher.__init__(self, topics if topics is not None else set())

        self.static_metadata = {}
        if static_metadata_files:
            for channel, metadata_filename in static_metadata_files.items():
                try:
                    with open(metadata_filename, 'r') as metadata_file:
                        self.static_metadata[channel] = json.load(metadata_file)
                except FileNotFoundError as e:
                    logging.error(f"Metadatafile '{metadata_filename}' does not exist! Skipping...")
                except json.JSONDecodeError as e:
                    logging.error(f"Metadatafile '{metadata_filename}' does not appear to be valid JSON! Skipping...")

        os.makedirs(self.output_dir, mode=0o755, exist_ok=True)

        self.channels = {}
        self.timestamps = {}
        self._init_file_writers()

        # Attach static metadata
        for address, metadata in self.static_metadata.items():
            if address in self.channels:
                self.channels[address].static_meta_info.update(metadata)

    def process(self, channel: str, data: int, sequence_id: int, timestamp: float) -> None:
        """
        Passes the received data sample and sequence_id to the channel specific file writer.

        :param channel:     DOOCS address of the channel from which `data` was received.
        :param data:        Read-only data sample from the previously subscribed to channel specified in `channel`.
        :param sequence_id: Sequence ID (macropulse number) of the data sample.
        :param timestamp:   Timestamp of the data sample.
        """
        if self._split_at_seq_id is not None and self._split_at_seq_id <= self._cur_seq_id:
            self._split_at_seq_id = None
            self._split()

        self._cur_seq_id = sequence_id

        try:
            self.channels[channel].store_sample(data, sequence_id)
            if self.record_timestamps:
                self.timestamps[channel].store_sample(timestamp, sequence_id)
        except (DuplicateDataSampleError, InconsistentDataSampleError) as e:
            logging.warning(f"Channel '{channel}': {str(e)}. Ignoring...")

        if self.split_size is not None and self.channels[channel].size >= self.split_size:
            self._split()

    def signal(self, event: Any) -> None:
        """
        Signal event handler. Schedules a data file split.

        :param event: Unused
        """

        if self._split_at_seq_id is None:
            self._split_at_seq_id = self._cur_seq_id + self.split_signal_delay

    def close(self) -> None:
        """
        Properly closes DataStreamFileWriter objects.
        """
        filepaths = []
        for channel in self.channels.values():
            filepaths.append(channel.filename)
            filepaths.append(channel.metadata_filename)
            channel.close()

        if self.record_timestamps:
            for timestamp in self.timestamps.values():
                filepaths.append(timestamp.filename)
                timestamp.close()

        self.publish(filepaths)


class RingFileWriter(FileWriter):
    """
    DxMAF module that writes data from subscribed channels to a ring-buffer like file and when receiving a trigger,
    closes that file, writes a corresponding metadata file and continues writing to a new file.
    """

    def _init_file_writers(self) -> None:
        """
        Initializes a _ring_ file writer for each channel.
        """

        self.tmp_ringfile_dirname = Path(tempfile.mkdtemp(dir=self.output_dir,
                                                          suffix=datetime.now().strftime(self.ring_dir_name)))

        for address in self._subscribed_channels:
            filepath = self.tmp_ringfile_dirname / (address.replace('/', '-') + '.npy')
            self.channels[address] = DataStreamRingFileWriter(filepath, self.ring_file_size,
                                                              memory_buffering=self._memory_buffering,
                                                              static_meta_info={'address': address})

            if self.record_timestamps:
                self.timestamps[address] = DataStreamRingFileWriter(filepath.with_suffix('.ts.npy'),
                                                                    self.ring_file_size,
                                                                    memory_buffering=self._memory_buffering,
                                                                    static_meta_info={'address': address,
                                                                                      'type': 'timestamp'})

    def _split(self) -> None:
        """
        Closes the current set of files and moves to the next (for example so that the previous data can be moved
        elsewhere). Publishes an event with the path to the closed ring directory.
        """
        old_ringfile_dirname = self.tmp_ringfile_dirname

        self.tmp_ringfile_dirname = Path(tempfile.mkdtemp(dir=self.output_dir,
                                                          suffix=datetime.now().strftime(self.ring_dir_name)))

        for channel_name in self.channels:
            filepath = self.tmp_ringfile_dirname / (channel_name.replace('/', '-') + '.npy')
            self.channels[channel_name].split(filepath)
            if self.record_timestamps:
                self.timestamps[channel_name].split(filepath.with_suffix('.ts.npy'))

        ringfile_dirname = datetime.now().strftime(self.ring_dir_name)
        os.rename(old_ringfile_dirname, ringfile_dirname)
        self.publish(ringfile_dirname)

    def __init__(self, channels: Set[str], ring_file_size: int, output_dir: PathLike = ".",
                 ring_dir_name: str = "ringbuffer_%Y-%m-%d_%H%M%S", record_timestamps: Optional[bool] = False,
                 split_signal_delay: Optional[int] = 1, memory_buffering: Optional[bool] = False,
                 static_metadata_files: Optional[Mapping[str, str]] = None, topics: Optional[Set[str]] = None):
        """
        Initializes the RingFileWriter object.

        :param channels:              Set (unique sequence) of DOOCS channel addresses for which `process` will be
                                      called in the event of new data.
        :param topics:                Topics to which to publish in the event of a split.
        :param ring_file_size:        Ring file size (in samples, per channel).
        :param output_dir:            Output directory path where the data files are stored. Allows for datestring
                                      tokens.
        :param ring_dir_name:         Ring subdirectory name format string. Allows for datestring tokens.
        :param split_signal_delay:    Delay in sequence id's after which a split signal is executed.
        :param memory_buffering:      Keep ring in memory until it is closed, then write to file system.
        :param static_metadata_files: Mapping of DOOCS channel addresses (same as `channels`) to files containing
                                      static metdata information.
        """
        self.ring_file_size = ring_file_size
        self.ring_dir_name = ring_dir_name
        self.ring_id = 0
        self._memory_buffering = memory_buffering

        super().__init__(channels, output_dir=output_dir, split_size=None, topics=topics,
                         record_timestamps=record_timestamps, split_signal_delay=split_signal_delay,
                         static_metadata_files=static_metadata_files)

        def close(self):
            for channel in self.channels.values():
                channel.close()
            if self.record_timestamps:
                for timestamp in self.timestamps.values():
                    timestamp.close()

            ringfile_dirname = datetime.now().strftime(self.ring_dir_name)
            os.rename(self.tmp_ringfile_dirname, ringfile_dirname)

            self.publish(ringfile_dirname)


# Export DxMAF modules
DXMAF_MODULES = (FileWriter, RingFileWriter)
