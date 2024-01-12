from typing import Set

from pubsub import pub


class EventPublisher:
    """Base class for plugins firing events for other modules to listen to."""

    def publish(self, event=None) -> None:
        """
        Send an event to all listeners on the given topics

        :param event: Object containing additional information about the event.
        :return:      None
        """

        for topic in self.topics:
            pub.sendMessage(topic, event=event)

    def __init__(self, topics: Set[str]):
        self.topics = topics
