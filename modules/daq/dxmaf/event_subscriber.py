from abc import ABCMeta, abstractmethod

from typing import Any


class EventSubscriber(metaclass=ABCMeta):
    """Abstract base class for plugins subscribing to data from DOOCS via ZMQ."""

    @abstractmethod
    def signal(self, event: Any) -> None:
        """
        Process event previously subscribed to.

        :param event: Object containing additional information about the event.
        :return:      None
        """
        pass
