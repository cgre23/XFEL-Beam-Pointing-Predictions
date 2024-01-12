import collections.abc
from functools import partial
import logging
import os
import shutil
import tempfile
import threading
import queue
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dxmaf.event_subscriber import EventSubscriber

SUFFIX_MAP = {
    '.zip':    'zip',
    '.tar':    'tar',
    '.tar.bz': 'bztar',
    '.tar.gz': 'gztar',
    '.tar.xz': 'xztar'
}


class FileMover(EventSubscriber):
    """
    DxMAF module that moves files to a new directory (optionally archived) when receiving a signal.
    """

    @staticmethod
    def worker(job_queue):
        while True:
            src, dst = job_queue.get()
            shutil.move(src, dst)
            logging.debug(f"Completed: Moving file '{src}' to '{dst}'.")
            job_queue.task_done()

    @staticmethod
    def archiveworker(job_queue):
        while True:
            dst, fmt, src = job_queue.get()

            commonpath = os.path.commonpath(src)
            if os.access(commonpath, os.W_OK):
                tmpdirdir = commonpath
            elif os.access('.', os.W_OK):
                tmpdirdir = '.'
            else:
                tmpdirdir = None

            with tempfile.TemporaryDirectory(dir=tmpdirdir) as tmpdirname:
                for filepath in src:
                    relfilepath = filepath.relative_to(commonpath)
                    tmpfilepath = tmpdirname / relfilepath
                    os.makedirs(tmpfilepath.parent, mode=0o755, exist_ok=True)
                    shutil.move(filepath, tmpfilepath)
                shutil.make_archive(dst, fmt, tmpdirname)

            logging.debug(f"Completed: Moving files '{src}' to '{dst}'.")
            job_queue.task_done()

    def signal(self, event: Any) -> None:
        """
        Moves the files specified in `event` to the configured destination.

        :param event: Object containing additional information about the event.
        """
        if not isinstance(event, collections.abc.Sequence) or isinstance(event, str):
            event = [event]

        filepaths = [Path(filepath) for filepath in event]

        logging.debug(f"Moving files '{[str(f) for f in filepaths]}' to '{self.destination}'.")
        if self.archive:
            archive_name = self.archive_name
            if self.archive_number > 0:
                archive_name += '-' + str(self.archive_number)
            while (archive_path := self.destination / archive_name).is_file():
                warnings.warn(f"File '{self.destination / self.archive_name}' unexpectedly exists! "
                              f"Incrementing sequence number.")
                self.archive_number += 1
                archive_name = self.archive_name + '-' + str(self.archive_number)
            self.job_queue.put((archive_path, self.archive_fmt, filepaths))
            self.archive_number += 1
        else:
            for filepath in filepaths:
                self.job_queue.put((filepath, self.destination))

    def __exit__(self):
        logging.info("Waiting for file transfers to complete.")
        self.job_queue.join()

    def __init__(self, destination: os.PathLike, archive: Optional[bool] = False):
        """
        Initializes the FileMover object.

        :param archive: If true, files are packaged into an archive. Archive format is derived from `destination` or
                        'zip'.
        :param destination: Relative or absolute path on the current file system where the files should be moved. Should
                            be a directory in case `archive` is false, and a file name otherwise.
        """

        self.archive = archive
        self.destination = Path(datetime.now().strftime(destination))
        self.archive_fmt = None
        self.archive_name = None
        self.archive_number = 0

        self.job_queue = queue.Queue()

        if self.archive:
            archive_formats = [format[0] for format in shutil.get_archive_formats()]

            # handle "double extension", ala '.tar.gz'
            suffix = self.destination.suffixes
            suffix = ''.join(suffix[-2:]) if suffix[-2:-1] == ['.tar'] else ''.join(suffix[-1:])

            self.destination = self.destination.with_name(self.destination.name[:-len(suffix)])

            if suffix and suffix in SUFFIX_MAP and SUFFIX_MAP[suffix] in archive_formats:
                self.archive_fmt = SUFFIX_MAP[suffix]
            elif suffix:
                warnings.warn(f"Suffix '{suffix}' is not a supported archiving format! Using default format.",
                              UserWarning)
                suffix = ''  # Next if sets default archive format
            if not suffix:
                if 'zip' in archive_formats:
                    self.archive_fmt = 'zip'
                else:
                    for fmt in archive_formats:
                        if fmt in SUFFIX_MAP.values():
                            self.archive_fmt = fmt

            assert self.archive_fmt is not None, "Didn't find any supported archive format library!"

            self.archive_name = self.destination.name
            self.destination = self.destination.parent
            logging.info(f"File archiving is enabled (format '{self.archive_fmt}').")

        logging.info(f"Will move files to destination '{destination}'.")

        os.makedirs(self.destination, mode=0o755, exist_ok=True)
        if not os.access(self.destination, os.W_OK):
            raise ValueError(f"Target directory '{destination}' is not writable.")

        threading.Thread(target=partial(FileMover.archiveworker if self.archive else FileMover.worker, self.job_queue),
                         daemon=True).start()


# Export DxMAF modules
DXMAF_MODULES = (FileMover,)
