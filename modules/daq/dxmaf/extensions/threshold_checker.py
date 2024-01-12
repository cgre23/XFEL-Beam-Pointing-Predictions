from typing import Set

import numpy

from dxmaf.data_subscriber import DataSubscriber
from dxmaf.event_publisher import EventPublisher


class ThresholdChecker(DataSubscriber, EventPublisher):
    """Event detector that checks data element-wise against a lower and upper threshold.

    Fires an event if either threshold is violated.
    """

    def process(self, channel: str, data, sequence_id: int, timestamp: float) -> None:
        """
        Checks a data sample from a previously subscribed to channel against configured lower and upper thresholds and
        fires an event in case a threshold is violated. Event will not fire again until a non-threshold-violating data
        sample has been processed.

        :param channel:     Address of the channel being processed.
        :param data:        Data sample from the channel with address `channel`.
        :param sequence_id: Sequence ID of the data sample with respect to other samples from the channel with
                            address `channel`.
        :param timestamp:   Timestamp of the data sample.
        :return:            None
        """

        data = numpy.asanyarray(data)   # no copy if already numpy array type

        if self.spectrum_mode:
            data = data[:, 1]

        if (data < self.lower_limit).any():
            self.publish(channel)
        elif (data > self.upper_limit).any():
            self.publish(channel)
        else:
            self.fired[channel] = False

    def publish(self, event=None) -> None:
        """
        Wraps EventPublisher's `publish` method to implement "debouncing" as described in `process` documentation.

        :param event: Object containing additional information about the event.
        """
        if not any(self.fired.values()):
            super().publish(event)

        self.fired[event] = True

    def __init__(self, channels: Set[str], topics: Set[str], lower_limit: float, upper_limit: float,
                 spectrum_mode: bool = False):
        """
        Initialize and configures the ThresholdChecker object.

        :param channels:      List of unique addresses to channels whose data is threshold checked.
        :param topics:        List of unique topic names to be notified when a threshold is violated.
        :param lower_limit:   The lower threshold. Set to -inf to disable.
        :param upper_limit:   The upper threshold. Set to inf to disable.
        :param spectrum_mode: Data is assumed to be Nx2, mapping a time-like axis to a value axis.
                              Thresholding is only performed on the value axis.
        """
        DataSubscriber.__init__(self, channels)
        EventPublisher.__init__(self, topics)

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.spectrum_mode = spectrum_mode
        self.fired = {channel: False for channel in channels}


# Export DxMAF modules
DXMAF_MODULES = (ThresholdChecker,)
