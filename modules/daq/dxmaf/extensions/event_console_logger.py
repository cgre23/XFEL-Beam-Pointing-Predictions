import logging
from typing import Any

from dxmaf.event_subscriber import EventSubscriber


class EventConsoleLogger(EventSubscriber):
    def signal(self, event: Any) -> None:
        """
        Writes a logging message at INFO level about the received signal.

        :param event: Object containing additional information about the event.
        :return:      None
        """

        logging.info(self.message)

    def __init__(self, message: str = "Signal Received!"):
        """
        Initializes the EventConsoleLogger object.

        :param message: The message to be logged when receiving a signal.
        """
        self.message = message


# Export DxMAF modules
DXMAF_MODULES = (EventConsoleLogger,)
