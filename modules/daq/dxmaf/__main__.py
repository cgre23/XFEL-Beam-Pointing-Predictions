import argparse
import collections
import dataclasses
import inspect
from itertools import islice, count
import logging
import pkgutil
import textwrap
import time
import typing
from datetime import datetime, timedelta
from os import PathLike
from time import sleep

import numpy as np
import pydoocs
import signal
import strictyaml
from pubsub import pub
from pytimeparse import parse as parse_timedelta_to_seconds
from multiprocessing import Event

from .data_subscriber import DataSubscriber
from .event_subscriber import EventSubscriber

__version__ = "1.1"

exit_evt = Event()

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.INFO)
logging.captureWarnings(True)


def termination_handler(sig, frame):
    exit_evt.set()
    print("\n\n")
    logging.info("Termination signal received.")


class Timedelta(strictyaml.ScalarValidator):
    def validate_scalar(self, chunk):
        try:
            return timedelta(seconds=parse_timedelta_to_seconds(chunk.contents))
        except ValueError:
            chunk.expecting_but_found("when expecting a timedelta")

    def to_yaml(self, data):
        if isinstance(data, timedelta):
            return str(data)
        raise strictyaml.exceptions.YAMLSerializationError(
            "expected a timedelta, got '{}' of type '{}'".format(data, type(data).__name__)
        )


# class DataClass(strictyaml.Map):
#     def __init__(self, dataclass):
#         if not dataclasses.is_dataclass(dataclass):
#             raise ValueError(f"'{dataclass}' is not a dataclass")
#
#         self.dataclass = dataclass
#         super().__init__({k: _resolve_strictyaml_validator(v)[0] for k, v in dataclass.__annotations__.items()})
#
#     def validate(self, chunk):
#         super().validate(chunk)
#         return self.dataclass(**chunk.contents)
#
#     def to_yaml(self, data):
#         super().to_yaml(data.asdict())


def _resolve_strictyaml_validator(annotation):
    type_hint_map = {
        bool:      strictyaml.Bool(),
        int:       strictyaml.Int(),
        float:     strictyaml.Float(),
        str:       strictyaml.Str(),
        datetime:  strictyaml.Datetime(),
        timedelta: Timedelta(),
        PathLike:  strictyaml.Str()
    }

    optional = False
    origin = typing.get_origin(annotation)
    if origin is not None:
        if origin is typing.Union:
            arg_subtypes = typing.get_args(annotation)
            validator, _ = _resolve_strictyaml_validator(arg_subtypes[0])
            optional = (arg_subtypes[-1] is type(None))
        elif issubclass(origin, collections.abc.Mapping):
            key_type, value_type = typing.get_args(annotation)
            validator = strictyaml.EmptyDict() | strictyaml.MapPattern(_resolve_strictyaml_validator(key_type)[0],
                                                                       _resolve_strictyaml_validator(value_type)[0])
        elif issubclass(origin, collections.abc.Sequence):
            value_type, = typing.get_args(annotation)
            validator = strictyaml.EmptyList() | strictyaml.Seq(_resolve_strictyaml_validator(value_type)[0])
        elif issubclass(origin, collections.abc.Set):
            value_type, = typing.get_args(annotation)
            validator = strictyaml.EmptyList() | strictyaml.UniqueSeq(_resolve_strictyaml_validator(value_type)[0])
        else:
            raise ValueError(f"Unknown typing format '{annotation}'!")
    else:
        if dataclasses.is_dataclass(annotation):
            validator = strictyaml.Map(
                {k: _resolve_strictyaml_validator(v)[0] for k, v in annotation.__annotations__.items()})
        else:
            validator = type_hint_map[annotation]

    return validator, optional


def _build_strictyaml_schema_from_signature(signature):
    schema = {}

    for param_name, param in signature.parameters.items():
        try:
            validator, optional = _resolve_strictyaml_validator(param.annotation)
        except ValueError as e:
            raise ValueError(f"Unknown typing format '{param}'! "
                             "If you think this is a valid type, please consider filing a bug report.")
        if param.default != inspect.Parameter.empty:
            schema[strictyaml.Optional(param_name, default=param.default)] = validator
        elif optional:
            schema[strictyaml.Optional(param_name, default=None)] = validator
        else:
            schema[param_name] = validator

    return strictyaml.Map(schema)


def _load_modules(extensions_path):
    modules = {}
    for module_finder, name, ispkg in pkgutil.iter_modules([extensions_path]):
        if ispkg:
            continue

        extension = module_finder.find_module(name).load_module(name)
        if hasattr(extension, 'DXMAF_MODULES'):
            for module in getattr(extension, 'DXMAF_MODULES'):
                if module.__name__ in modules:
                    raise RuntimeError(f"duplicate DxMAF module '{extension}'")
                try:
                    modules[module.__name__] = getattr(extension, module.__name__)
                except AttributeError:
                    raise AttributeError(f"extension '{name}' exports unknown DxMAF module '{module}'")

    return modules


def main():
    # Command Line Arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
                    Python DOOCS eXtensible Middlelayer Application Framework
                    
                    DESCRIPTION
                    -----------
                    Used to build custom middle-layer services using Python from a library of modules. Refer to example
                    configuration file for more information on available modules.
                    
                    Command line arguments take precedence over options specified in config file.
                    """),
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config', default='run.conf',
                        help="Path to the configuration file specifying the module architecture for the service.")
    parser.add_argument('-n', '--name',
                        help="Optional name for the service to be used by logs and modules.")
    parser.add_argument('-p', '--rpc-polling-period', default='0.1',
                        help="Period (in seconds) of data polling for channels that cannot be subscribed to via ZMQ. "
                             "Setting to 0 ignores RPC channels.")
    parser.add_argument('--force-rpc', action='store_true',
                        help="Disable ZMQ subscription and get all channels via DOOCS RPC calls.")
    parser.add_argument('--rpc-divider', type=int, default=1,
                        help="<EXPERIMENTAL> Divider for the machine trigger to obtain the DOOCS RPC readout trigger.")
    parser.add_argument('-t', '--stop-time', nargs='+', metavar=('HH:MM:SS', 'DD-MM-YYYY'),
                        help="Time and optionally date at which to stop.")
    parser.add_argument('-d', '--duration', metavar='HH:MM:SS',
                        help="Duration for how long to run.")
    parser.add_argument('--timeout',
                        type=lambda x: int(x) if int(x) > 0 else argparse.ArgumentError(
                            f"'{x}' is not a positive integer"),
                        help="Number of seconds after which a timeout occurs while waiting for data.")
    loglevel_group = parser.add_mutually_exclusive_group()
    loglevel_group.add_argument('-v', '--verbose', action='store_true',
                                help="Print additional information to the standard output (stdout) "
                                     "and error output (stderr).")
    loglevel_group.add_argument('-q', '--quiet', action='store_true',
                                help="Suppress all non-error / non-warning output.")

    cli_args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)

    # Load config file
    config_schema = strictyaml.Map({
        strictyaml.Optional('extensions_path', default='./extensions'): strictyaml.Str(),
        strictyaml.Optional('stop_time'):                               strictyaml.Datetime(),
        strictyaml.Optional('duration'):                                Timedelta(),
        strictyaml.Optional('timeout', default=2):                      strictyaml.Float(),
        strictyaml.Optional('rpc_polling_period', default=0.1):         strictyaml.Float(),
        'application':                                                  strictyaml.Seq(
            strictyaml.Map({
                'type':                          strictyaml.Str(),
                strictyaml.Optional('channels'): strictyaml.Seq(strictyaml.Str()),
                strictyaml.Optional('topics'):   strictyaml.Seq(strictyaml.Str()) | strictyaml.EmptyList(),
                strictyaml.Optional('args'):     strictyaml.MapPattern(strictyaml.Str(), strictyaml.Any())
            }))
    })

    with open(cli_args.config, 'r') as f:
        config = strictyaml.load(f.read(), config_schema)

    # Update config with command line arguments
    if cli_args.name is not None:
        config['name'] = cli_args.name
    if cli_args.force_rpc:
        config['force_rpc'] = True
    if cli_args.timeout is not None:
        config['timeout'] = cli_args.timeout
    if cli_args.rpc_polling_period is not None:
        config['rpc_polling_period'] = cli_args.rpc_polling_period
    if cli_args.stop_time is not None:
        if len(cli_args.stop_time) > 1:
            config['stop_time'] = datetime.strptime(cli_args.stop_time.join(' '), '%H:%M:%S %d-%m-%Y')
        else:
            config['stop_time'] = datetime.combine(datetime.now().date(),
                                                   datetime.strptime(cli_args.stop_time[0], '%H:%M:%S').time())
    if cli_args.duration is not None:
        config['duration'] = timedelta(seconds=parse_timedelta_to_seconds(cli_args.duration))

    # Set Logging Level
    if cli_args.quiet:
        logging.root.setLevel(logging.WARNING)
    elif cli_args.verbose:
        logging.root.setLevel(logging.DEBUG)

    # Load module classes from ./module directory.
    available_modules = _load_modules(config['extensions_path'].data)
    logging.info(f"Loaded modules {tuple(available_modules.keys())}.")

    # Build application from modules according to configuration file. Remember all subscribed to DOOCS channels.
    app = []
    channel_subscribers = {}
    for modulespec in config['application']:
        if modulespec['type'].data not in available_modules:
            raise RuntimeError(f"Module '{modulespec['type']}' not found.")
        cls = available_modules[modulespec['type'].data]

        if 'args' in modulespec:
            signature = inspect.signature(cls.__init__)
            signature = signature.replace(
                parameters=[param for param in signature.parameters.values() if param.name != 'self']
            )
            if issubclass(cls, DataSubscriber):
                signature = signature.replace(
                    parameters=[param for param in signature.parameters.values() if param.name != 'channels']
                )
            schema = _build_strictyaml_schema_from_signature(signature)
            modulespec['args'].revalidate(schema)
            module_args = modulespec['args'].data
        else:
            module_args = {}

        if issubclass(cls, DataSubscriber) and 'channels' in modulespec:
            module = cls(channels=modulespec['channels'].data, **module_args)
            for channel in modulespec['channels'].data:
                channel_subscribers.setdefault(channel, []).append(module)
        else:
            module = cls(**module_args)

        if issubclass(cls, EventSubscriber) and 'topics' in modulespec and modulespec['topics'].data:
            pub.subscribe(module.signal, modulespec['topics'].data)

        app.append(module)

    # Check ZMQ support
    zmq_channels = set()
    if not cli_args.force_rpc:
        addr_it = iter(channel_subscribers.keys())
        while addr_chunk := list(islice(addr_it, 64)):  # Check 64 properties at a time
            res = pydoocs.connect(addr_chunk, checkonly=True)
            for addr in addr_chunk:
                try:
                    if res[addr]['error'] == 'OK':
                        zmq_channels.add(addr)
                except KeyError:
                    logging.warning(f"Failed to check ZeroMQ support for address '{str(addr)}', falling back to RPC.")

    rpc_channels = channel_subscribers.keys() - zmq_channels

    zmq_channels = list(zmq_channels)
    if len(zmq_channels) > 64:
        logging.warning("Requested more than 64 ZMQ channels, falling back to RPC for remaining channels.")
        rpc_channels.update(zmq_channels[64:])
        zmq_channels = zmq_channels[:64]

    logging.info("Channels collected via ZMQ:\n" + '\n'.join(zmq_channels))
    logging.info("Channels collected via RPC:\n" + '\n'.join(rpc_channels))

    # Connect to DOOCS channels.
    if zmq_channels:
        pydoocs.connect(zmq_channels,
                        cycles=-1,
                        bufs=8 * len(zmq_channels))
    else:
        pydoocs.connect(['XFEL.DIAG/TIMINGINFO/MACROPULSENUMBER/MACRO_PULSE_NUMBER'], cycles=-1)

    # Compute correct stop time
    stop_time = None
    if 'duration' in config:
        stop_time = datetime.now() + config['duration'].data
    if 'stop_time' in config:
        if config['stop_time'].data < datetime.now():
            config['stop_time'] = datetime.combine(datetime.today(), config['stop_time'].data.time())
            if config['stop_time'].data < datetime.now():
                config['stop_time'] = config['stop_time'].data + timedelta(days=1)

    # If duration and stop time specified, take the latest stop time.
    if 'stop_time' in config:
        if stop_time is None or (stop_time is not None and stop_time > config['stop_time'].data):
            stop_time = config['stop_time'].data

    logging.info(f"Starting at {datetime.now()}. "
                 f"Stopping at {stop_time if stop_time is not None else '... never'}.")

    last_data_timestamp = time.time()
    for i in count():
        if exit_evt.is_set():
            break

        if time.time() - last_data_timestamp > config['timeout'].data:
            logging.error(f"Timeout after waiting for data for {config['timeout'].data} seconds.")
            break
        elif stop_time is not None and stop_time < datetime.now():
            logging.debug(f"Stop time reached at {datetime.now()}.")
            break

        sleep(0.04)

        response = pydoocs.getdata()
        if response is None:
            continue

        for channel in response:
            logging.debug(f"Received id '{channel['macropulse']}' for channel '{channel['name']}'.")

            if isinstance(channel['data'], np.ndarray):
                channel['data'].flags.writeable = False
            for subscriber in channel_subscribers.get(channel['name'], ()):
                subscriber.process(channel['name'], channel['data'], channel['macropulse'], channel['timestamp'])
        if i % cli_args.rpc_divider == 0:
            for channel in rpc_channels:
                try:
                    res = pydoocs.read(channel)
                except pydoocs.DoocsException:
                    logging.warning(f"Failed to read channel '{channel}' via RPC!")
                    continue

                logging.debug(f"Received id '{res['macropulse']}' for channel '{channel}' (via RPC).")

                if isinstance(res['data'], np.ndarray):
                    res['data'].flags.writeable = False
                for subscriber in channel_subscribers[channel]:
                    subscriber.process(channel, res['data'], response[0]['macropulse'], res['timestamp'])

        last_data_timestamp = time.time()

    if not exit_evt.is_set():
        logging.info("Break condition met. Stopping.")

    for module in app:
        if isinstance(module, DataSubscriber):
            module.close()
        try:
            module.__exit__()
        except AttributeError:
            pass

    pydoocs.disconnect()


if __name__ == '__main__':
    main()
