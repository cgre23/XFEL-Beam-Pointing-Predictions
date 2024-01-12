import logging
import subprocess
from typing import Any

from dxmaf.event_subscriber import EventSubscriber

logging.getLogger(__name__).addHandler(logging.NullHandler())


class CommandRunner(EventSubscriber):
    """
    DxMAF module that asynchronously runs a specified command when receiving a signal.
    """

    def signal(self, event: Any) -> None:
        """
        Runs the configured command.

        :param event: List of strings to be used for token substitution in the command string if token substitution is
                      enabled.
        """

        command = self.command
        if self.substitute:
            command = command.format(event)

        logging.info(f"Executing command '{command}'.")
        subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def __init__(self, command: str, substitute: bool = False):
        """
        Initializes the CommandRunner object.

        :param command:    Command to run when receiving a signal.
        :param substitute: Enables replacement of {0}, {1}, ... tokens in the command string with elements from the
                           event information object. See `signal`.
        """
        # self.command = shlex.split(self.command)
        self.command = command
        self.substitute = substitute


# Export DxMAF modules
DXMAF_MODULES = (CommandRunner,)
