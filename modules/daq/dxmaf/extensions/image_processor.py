import json
import logging
import os
from datetime import datetime
from typing import Set, Optional
import numpy as np
import pydoocs
from dxmaf.data_subscriber import DataSubscriber
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

doocs_write = 1


class ImageProcessor(DataSubscriber):
    """
    DxMAF module that processes images from subscribed channels and writes to DOOCS channels.
    """

    def __init__(self, channels: Set[str], SASE: str, output_file: Optional[str] = None):
        """
        Initializes the ModelPredictor object.

        :param channels: Set of DOOCS channel addresses for which `process` will be called in the event of new data.
        """
        DataSubscriber.__init__(self, channels)
        self.channels = channels
        self.roi = None  # initialize region of interest flag
        self.sigma = 5
        self.SASE = SASE
        print('Number of channels:', len(channels))

    def gaussian(self, x, a, b, c):
        val = a * np.exp(-(x - b)**2 / (2*c**2))
        return val

    def resample_rows_columns(self, a):
        cols = a.mean(axis=0)
        cols_idx = np.linspace(0, len(cols), num=len(cols), endpoint=False)

        rows = a.mean(axis=1)
        rows_idx = np.linspace(0, len(rows), num=len(rows), endpoint=False)
        # Resample rows and columns
        cols_xfine = np.linspace(0, len(cols), num=(len(cols)) * 10, endpoint=False)
        rows_xfine = np.linspace(0, len(rows), num=(len(rows)) * 10, endpoint=False)

        cols_zero = cols - np.mean(cols[0:10])
        rows_zero = rows - np.mean(rows[0:10])
        return cols_idx, rows_idx, cols_xfine, rows_xfine, cols_zero, rows_zero

    def process(self, channel: str, data, sequence_id: int, timestamp: float) -> None:
        """
        Process data from a channel previously subscribed to.

        :param channel: DOOCS address of the channel from which `data` was received.
        :param data: Read-only data sample from the previously subscribed to channel specified in `channel`.
        :param sequence_id: Sequence ID (macropulse number) of the data sample.
        :param timestamp: Timestamp of the data sample.
        :return: None
        """
        acquisition_signal = pydoocs.read(channel.replace('BEAMVIEW.RAW', 'STATE'))["data"]

        if acquisition_signal == 'ACQUIRING':
            cols_idx, rows_idx, cols_xfine, rows_xfine, cols_zero, rows_zero = self.resample_rows_columns(data)

            try:
                cols_popt, pcov = curve_fit(self.gaussian, cols_idx, cols_zero, p0=[max(cols_zero), np.mean(cols_idx), np.std(cols_idx)])
                rows_popt, pcov_r = curve_fit(self.gaussian, rows_idx, rows_zero, p0=[max(rows_zero), np.mean(rows_idx), np.std(rows_idx)])
                row_gaussian = self.gaussian(rows_xfine, *rows_popt)
                col_gaussian = self.gaussian(cols_xfine, *cols_popt)
            except Exception as e:
                logging.error('Not able to fit Gaussian curve to the intensity plot. Error: %s', str(e))
                return

            if cols_popt[0] > 10 and rows_popt[0] > 10:
                if self.roi is None:  # ROI is only calculated at the first iteration
                    com_x = int(cols_popt[1])
                    com_y = int(rows_popt[1])
                    beamsize_x = int(abs(cols_popt[2]))
                    beamsize_y = int(abs(rows_popt[2]))

                    # SRA condition
                    if beamsize_x > 100:
                        beamsize_x = 70
                        sigma = 3
                    if beamsize_y > 100:
                        beamsize_y = 50
                        sigma = 3

                    self.roi = [com_y - beamsize_y * self.sigma, com_y + beamsize_y * self.sigma,
                                com_x - beamsize_x * self.sigma, com_x + beamsize_x * self.sigma]

                cropped_a = data[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                cols_idx, rows_idx, cols_xfine, rows_xfine, cols_zero, rows_zero = self.resample_rows_columns(cropped_a)

                try:
                    cols_popt, pcov = curve_fit(self.gaussian, cols_idx, cols_zero,
                                                p0=[max(cols_zero), np.mean(cols_idx), np.std(cols_idx)])
                    rows_popt, pcov_r = curve_fit(self.gaussian, rows_idx, rows_zero,
                                                 p0=[max(rows_zero), np.mean(rows_idx), np.std(rows_idx)])
                    row_gaussian = self.gaussian(rows_xfine, *rows_popt)
                    col_gaussian = self.gaussian(cols_xfine, *cols_popt)
                except Exception as e:
                    logging.error('Not able to fit Gaussian curve to the cropped intensity plot. Error: %s', str(e))

                com_x = cols_popt[1] + self.roi[2]
                com_y = rows_popt[1] + self.roi[0]
                beamsize_x = abs(cols_popt[2])
                beamsize_y = abs(rows_popt[2])
                skewness_y = skew(rows_zero)
                kurtosis_y = kurtosis(rows_zero, fisher=False)
                skewness_x = skew(cols_zero)
                kurtosis_x = kurtosis(cols_zero, fisher=False)
                max_intensity = np.max(cropped_a)
                if max_intensity > 5000:
                    logging.info('Beam intensity saturated....add filter/attenutator.')
                rmse_x = np.sqrt(np.sum(np.square(cols_zero - col_gaussian[::10])) / len(cols_zero))
                rmse_y = np.sqrt(np.sum(np.square(rows_zero - row_gaussian[::10])) / len(rows_zero))
                fit_error_x = cols_popt[0] - max(cols_zero)
                fit_error_y = rows_popt[0] - max(rows_zero)

                if doocs_write == 1:
                    self.write_to_doocs(com_x, com_y, beamsize_x, beamsize_y, skewness_x, skewness_y, kurtosis_x,
                                         kurtosis_y, fit_error_x, fit_error_y, rmse_x, rmse_y, max_intensity)

            else:
                # logging.error('Intensity too low.')
                pass

        else:
            # logging.error('No imager acquisition')
            pass

    def write_to_doocs(self, com_x, com_y, beamsize_x, beamsize_y, skewness_x, skewness_y, kurtosis_x, kurtosis_y,
                       fit_error_x, fit_error_y, rmse_x, rmse_y, max_intensity):
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/COM_X_MEASUREMENT', com_x)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/COM_Y_MEASUREMENT', com_y)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/BEAMSIZE_X_MEASUREMENT', beamsize_x)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/BEAMSIZE_Y_MEASUREMENT', beamsize_y)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/SKEWNESS_X_MEASUREMENT', skewness_x)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/SKEWNESS_Y_MEASUREMENT', skewness_y)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/KURTOSIS_X_MEASUREMENT', kurtosis_x)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/KURTOSIS_Y_MEASUREMENT', kurtosis_y)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/FIT_ERROR_X', fit_error_x)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/FIT_ERROR_Y', fit_error_y)
        #pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/FIT_ERROR_X_SQUARED', rmse_x)
        #pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/FIT_ERROR_Y_SQUARED', rmse_y)
        pydoocs.write(f'XFEL.UTIL/DYNPROP/BEAM_PREDICT.{self.SASE}/MAX_INTENSITY_ON_SCREEN', max_intensity)

    def close(self) -> None:
        """
        Save data to file when finished or session is interrupted.
        """
        pass


# Export DxMAF modules
DXMAF_MODULES = (ImageProcessor,)
