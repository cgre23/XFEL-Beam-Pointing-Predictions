# 
# Based on information from https://confluence.desy.de/display/MCS/Timing+Pattern+for+DOOCS+Servers
# 
# EXAMPLE
# import numpy as np
# import matplotlib.pyplot as plt
# from mcxdaqtools.DaqTimingPatter import parse_timing_pattern_packed
# 
# with h5py.File('sample.hdf5', 'r') as file:
#   bunch_pattern = unpack_timing_pattern(file['XFEL.DIAG/TIMINGINFO/TIME1.BUNCH_PATTERN/Value'][0])
#   plt.figure()
#   plt.plot(bunch_pattern.destination[::4])
#   plt.yticks([e.value for e in DestinationXfel], [e.name for e in DestinationXfel])


from enum import IntEnum, IntFlag, unique
import numpy as np
from ctypes import LittleEndianStructure, c_float, c_uint8, c_uint16, POINTER, cast

LINACS = ('xfel', 'flash')


@unique
class BunchChargeSetting(IntEnum):
    NO_BUNCH = 0
    pC_20 = 1
    pC_30 = 2
    pC_40 = 3
    pC_60 = 4
    pC_90 = 5
    pC_130 = 6
    pC_180 = 7
    pC_250 = 8
    pC_360 = 9
    pC_500 = 10
    pC_710 = 11
    pC_1000 = 12
    pC_1420 = 13
    pC_2000 = 14
    UNLIMITED = 15


@unique
class DestinationXfel(IntEnum):
    NONE = 0
    LASER_STAND_ALONE = 1
    T5D = 2
    G1D = 3
    T4D = 4
    I1D = 5
    B1D = 6
    B2D = 7
    TLD = 8


@unique
class DestinationFlash(IntEnum):
    NONE = 0
    FLASH2 = 2
    FLASH1 = 4
    FLASH_FORWARD = 8


Destination = {'xfel': DestinationXfel, 'flash': DestinationFlash}


@unique
class InjectorLaserTriggersXfel(IntFlag):
    INJECTOR_1_LASER_1 = 1
    INJECTOR_1_LASER_2 = 2
    INJECTOR_2_LASER_1 = 4
    INJECTOR_2_LASER_2 = 8


@unique
class InjectorLaserTriggersFlash(IntFlag):
    INJECTOR_LASER_1 = 1
    INJECTOR_LASER_2 = 2
    INJECTOR_LASER_3 = 4


InjectorLaserTriggers = {'xfel': InjectorLaserTriggersXfel, 'flash': InjectorLaserTriggersFlash}


@unique
class SeedUserLaserTriggersXfel(IntFlag):
    SEED_USER_LASER_1 = 1
    SEED_USER_LASER_2 = 2
    SEED_USER_LASER_3 = 4
    SEED_USER_LASER_4 = 8
    SEED_USER_LASER_5 = 16
    SEED_USER_LASER_6 = 32


@unique
class SeedUserLaserTriggersFlash(IntFlag):
    SEED_LASER_1 = 1
    SEED_LASER_2 = 2
    PUMP_PROBE_LASER_1 = 8
    PUMP_PROBE_LASER_2 = 16


SeedUserLaserTriggers = {'xfel': SeedUserLaserTriggersXfel, 'flash': SeedUserLaserTriggersFlash}


@unique
class SpecialFlagsXfel(IntFlag):
    SPECIAL_REP_RATE = 1
    BEAM_DISTRIBUTION_KICKER = 16
    TLD_SOFT_KICK = 32
    WIRE_SCANNER = 64
    TDS_BC_2 = 128
    TDS_BC_1 = 256
    TD_KICKER_INJECTOR_1 = 512


@unique
class SpecialFlagsFlash(IntFlag):
    SPECIAL_REP_RATE = 1
    PHOTON_MIRROR = 32
    LOLA = 128
    CRISP_KICKER = 256


SpecialFlags = {'xfel': SpecialFlagsXfel, 'flash': SpecialFlagsFlash}


class TimingWordPacked(LittleEndianStructure):
    _fields_ = [
        ('bunch_charge_setting', c_uint8, 4),
        ('injector_laser_triggers', c_uint8, 4),
        ('seed_user_laser_triggers', c_uint8, 6),
        ('reserved_1', c_uint8, 2),
        ('reserved_2', c_uint8, 2),
        ('destination', c_uint8, 4),
        ('special_flags', c_uint16, 10),
    ]


timing_word_type = {linac: np.dtype([('bunch_charge_setting', BunchChargeSetting),
                                     ('injector_laser_triggers', InjectorLaserTriggers[linac]),
                                     ('seed_user_laser_triggers', SeedUserLaserTriggers[linac]),
                                     ('destination', Destination[linac]),
                                     ('special_flags', SpecialFlags[linac])]) for linac in LINACS}


def unpack_timing_word(timing_word):
    timing_word = np.asarray(timing_word)
    if timing_word.dtype.itemsize != 4:
        raise TypeError("`timing_word` must have 4-byte (word) datatype convertible to numpy scalar")

    timing_word_bitfield = cast(timing_word.ctypes.data_as(POINTER(c_float)),
                                POINTER(TimingWordPacked)).contents

    return np.asarray((
        timing_word_bitfield.bunch_charge_setting,
        timing_word_bitfield.injector_laser_triggers,
        timing_word_bitfield.seed_user_laser_triggers,
        timing_word_bitfield.destination,
        timing_word_bitfield.special_flags), dtype=np.uint16)


def unpack_timing_pattern(pattern):
    pattern = np.asarray(pattern)
    if pattern.dtype.itemsize != 4:
        raise TypeError("`pattern` must have 4-byte (word) datatype convertible to numpy array")

    # There can be many more bunches than bunch codes,
    # so compute them once and then produce the output array
    unique_timing_words = np.unique(pattern)
    timing_word_lut = np.vstack([unpack_timing_word(word) for word in unique_timing_words])

    pattern_unpacked = np.zeros(pattern.shape + (5,), dtype=np.uint16)

    it = np.nditer(pattern, flags=['multi_index'])
    for word in it:
        if word != 0:
            pattern_unpacked[it.multi_index] = timing_word_lut[unique_timing_words == word]

    return pattern_unpacked


def decode_timing_pattern(value, linac=LINACS[0]):
    raise NotImplementedError('TODO')

    # if not isinstance(linac, str) or linac.lower() not in LINACS:
    #     raise TypeError(f"`mode` must be either of {LINACS}")
    # TODO
    # patterns_unpacked.flat[i]['bunch_charge_setting'] = BunchChargeSetting(pattern.bunch_charge_setting)
    # patterns_unpacked.flat[i]['injector_laser_triggers'] = InjectorLaserTriggers[linac](
    #     pattern.injector_laser_triggers)
    # patterns_unpacked.flat[i]['seed_user_laser_triggers'] = SeedUserLaserTriggers[linac](
    #     pattern.seed_user_laser_triggers)
    # patterns_unpacked.flat[i]['destination'] = Destination[linac](pattern.destination)
    # patterns_unpacked.flat[i]['special_flags'] = SpecialFlags[linac](pattern.special_flags)

