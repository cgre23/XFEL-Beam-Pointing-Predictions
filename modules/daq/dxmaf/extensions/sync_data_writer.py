import json
import logging
import os
import pandas as pd
import tempfile
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Set, Any, Optional, Mapping
import numpy
import pydoocs
import torch
import torch.nn as nn
from dxmaf.data_subscriber import BufferedDataSubscriber
from collections import namedtuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TaggedData = namedtuple('TaggedData', ['sequence_id', 'data'])


class SyncDataWriter(BufferedDataSubscriber):
    """
    DxMAF module that writes data from subscribed channels to numpy and metadata files.

    # TODO: Add file system free space check.
    """

    def __init__(self, channels: Set[str], output_file: Optional[str] = None,):
        """
        Initializes the ModelPredictor object.

        :param channels:              Set (unique sequence) of DOOCS channel addresses for which `process` will be
                                      called in the event of new data.
        """

        BufferedDataSubscriber.__init__(self, channels, len(channels))
        self.channels = channels
        print('Number of channels:', len(channels))
        

        export_columns = self.channels.copy()
        self.df_export = pd.DataFrame(columns=export_columns)
        self.output_file = Path(datetime.now().strftime(output_file))
        
    def process_incomplete(self, dataset: Mapping[str, numpy.ndarray], sequence_id: int) -> None:
        #logging.info('Process Incomplete')
        pass
        
    def process_complete(self, dataset: Mapping[str, float], sequence_id: int) -> None:
        self.df_export.loc[sequence_id]=dataset

    def process(self, channel: str, data: int, sequence_id: int, timestamp: float) -> None:
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
        if 'XGM/XGM' in channel:
                if len(data) > 0:
                        data = data[0][1]
                else:
                        data = 0
        
        highest_sequence_id = max([*self.buffer.keys(), sequence_id])
        if sequence_id < (highest_sequence_id - self.max_buffer_size):
            return

        for key in list(self.buffer.keys()):
            if key < (highest_sequence_id - self.max_buffer_size):
                self.process_incomplete(self.buffer[key], key)
                del self.buffer[key]

        self.buffer.setdefault(sequence_id, {})[channel] = data

        if len(self.buffer[sequence_id]) == len(self.channels):
            self.process_complete(self.buffer[sequence_id], sequence_id)
            del self.buffer[sequence_id]
       

    def close(self) -> None:
        """
        Save data to file when finished or session is interrupted.

        """
        
        logging.info('Writing to file....')
        try:
            self.df_export.to_parquet(self.output_file, compression='gzip')
            logging.info('Data saved successfully to: %s', self.output_file)
        except:
            logging.info('Failed to save data to: %s', self.output_file)
        



# Export DxMAF modules
DXMAF_MODULES = (SyncDataWriter,)
