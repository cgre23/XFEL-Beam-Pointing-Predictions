import sys
import string
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QFileSystemModel
from gui.UIdaq import Ui_Form
import threading, queue
import shutil
import re
from datetime import datetime, timedelta
from modules.spectr_gui import send_to_desy_elog
import gui.resources_rc
import subprocess
import time
do_doocs = 1
if do_doocs == 1:
    import pydoocs
import yaml

# Define status icons (available in the resource file built with "pyrcc5"
ICON_RED_LED = ":/icons/led-red-on.png"
ICON_GREEN_LED = ":/icons/green-led-on.png"
ICON_BLUE_LED = ":/icons/blue-led-on.png"
ICON_GREY_LED = ":/icons/grey-led-on.png"
ICON_ORANGE_LED = ":/icons/orange-led-on.png"
ICON_PURPLE_LED = ":/icons/purple-led-on.png"
ICON_YELLOW_LED = ":/icons/yellow-led-on.png"

class DAQApp(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        # Initialize parameters
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.logstring = []
        self.sa1_sequence_prefix = 'XFEL.UTIL/TASKOMAT/DAQ_SA1'
        self.sa1_sequence_background_step =self.sa1_sequence_prefix +'/STEP007'
        self.config_path = 'modules/daq/docs/'
        self.data_path = 'modules/daq/runs/'
        self.model_path = 'modules/models/'
        self.ui.sequence_button.setCheckable(True)
        self.ui.sequence_button.setEnabled(True)
        self.ui.sequence_button.clicked.connect(self.toggleSequenceButton)
        self.ui.fetch_button.clicked.connect(self.fetch_doocs_data)
        self.ui.write_button.clicked.connect(self.write_doocs_data)
        self.ui.trainmodel_button.clicked.connect(self.train_model)
        self.ui.launchmodel_button.clicked.connect(self.launch_model)
        self.ui.restart_image_button.clicked.connect(self.restart_image_analysis_server)
        self.ui.restart_prediction_button.clicked.connect(self.restart_model_prediction_server)
        self.ui.restart_training_button.clicked.connect(self.restart_model_training_server)
        self.ui.measurement_time.valueChanged.connect(self.update_estimated_time)
        self.ui.iterations.valueChanged.connect(self.update_estimated_time)
        self.ui.warning.setStyleSheet("""QLabel { color: red;}""")
        self.ui.trainmodel_button.setEnabled(False)
        self.ui.launchmodel_button.setEnabled(False)
        self.list_directories()
        self.check_crls()

    def toggleSequenceButton(self):
        # if button is checked
        if self.ui.sequence_button.isChecked():
            # setting background color to blue
            self.palette = self.ui.sequence_button.palette()
            self.palette.setColor(QtGui.QPalette.Button, QtGui.QColor('blue'))
            self.ui.sequence_button.setPalette(self.palette)
            self.ui.sequence_button.setText("Force Stop DAQ")

            t = threading.Thread(target=self.start_sequence)
            t.daemon = True
            t.start()
            #self.start_sequence()
            #self.logbooktext = ''.join(self.logstring)
            #self.logbook_entry(self.logbooktext)
            #self.palette.setColor(QtGui.QPalette.Button, QtGui.QColor('white'))
            #self.ui.sequence_button.setPalette(self.palette)
            #self.ui.sequence_button.setText("Start DAQ")

        # if it is unchecked
        else: # Force Stop
            # set background color back to white
            self.palette = self.ui.sequence_button.palette()
            self.palette.setColor(QtGui.QPalette.Button, QtGui.QColor('white'))
            self.ui.sequence_button.setPalette(self.palette)
            self.ui.sequence_button.setText("Start DAQ")
            # Force Stop sequence
            try:

                pydoocs.write(self.sa1_sequence_prefix+'/FORCESTOP', 1)
                stop_log = datetime.now().isoformat(' ', 'seconds')+': Aborted the Taskomat sequence.\n'
                stop_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Aborted the Taskomat sequence.  <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
                self.logstring.append(stop_log)
                self.ui.textBrowser.append(stop_log_html)
                # Write to logbook
                self.logbooktext = ''.join(self.logstring)
                self.logbook_entry(widget=self.tab, text=self.logbooktext)
            except:
                print('Not able to stop the sequence.\n')
                stop_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Not able to stop the sequence.  <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
                self.ui.textBrowser.append(stop_log_html)

    def start_dxmaf(self):
        """ Start DXMAF measurement """
        self.proc = subprocess.Popen(["/bin/sh",  "./modules/daq/launch_writer_1.sh"])


    def start_sequence(self):
        """ Start DAQ measurement """
        try:
            self.set_config() # make sure DXMAF config file has the correct measurement duration
            pydoocs.write(self.sa1_sequence_prefix+'/RUN.ONCE', 1)
            self.logstring = []
            start_log = datetime.now().isoformat(' ', 'seconds')+': Started Taskomat sequence.\n'
            start_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Started the Taskomat sequence.  <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
            self.logstring.append(start_log)
            self.ui.textBrowser.append(start_log_html)
            undulators =  ''.join(filter(str.isdigit, self.ui.SASEoptions.currentText()) )
            dxmaf_flag = True
            while pydoocs.read(self.sa1_sequence_prefix+'/RUNNING')['data'] ==1: #############
                # Start running dxmaf only when Step 7 is running and only call the start function once.
                if pydoocs.read(self.sa1_sequence_background_step+'.RUNNING')['data'] == 1:  #################
                    if dxmaf_flag == True:
                    	dxmaf_flag = False
                    	now = datetime.now()
                    	dt_string = now.strftime("%Y-%m-%d")
                    	path = self.data_path + 'SA1/' + dt_string
                    	self.makedirs(path)
                    	self.start_dxmaf()


                log = pydoocs.read(self.sa1_sequence_prefix+'/LOG.LAST')['data']
                if log not in self.ui.textBrowser.toPlainText():
                    self.ui.textBrowser.append(pydoocs.read(self.sa1_sequence_prefix+'/LOG_HTML.LAST')['data'])
                    self.logstring.append(pydoocs.read(self.sa1_sequence_prefix+'/LOG.LAST')['data']+'\n')
                    time.sleep(0.001)
                 #pass
            self.update_taskomat_logs()
            self.ui.sequence_button.setChecked(False)
            self.palette = self.ui.sequence_button.palette()
            self.palette.setColor(QtGui.QPalette.Button, QtGui.QColor('white'))
            self.ui.sequence_button.setPalette(self.palette)
            self.ui.sequence_button.setText("Start DAQ")
            self.logbooktext = ''.join(self.logstring)
            self.logbook_entry(self.logbooktext)
        except:
            print('Not able to start Taskomat sequence.')
            start_log = datetime.now().isoformat(' ', 'seconds')+': Not able to start Taskomat sequence.\n'
            start_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Not able to start Taskomat sequence.  <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
            self.logstring.append(start_log)
            self.ui.textBrowser.append(start_log_html)
            #self.ui.sequence_button.setChecked(False)
            self.ui.sequence_button.setText("Start DAQ")


    def update_taskomat_logs(self):
        """" Get the last log from DOOCS and append to string """
        self.last_log_html = pydoocs.read(self.sa1_sequence_prefix+'/LOG_HTML.LAST')['data']
        self.last_log = pydoocs.read(self.sa1_sequence_prefix+'/LOG.LAST')['data']
        self.logstring.append(self.last_log+'\n')
        self.ui.textBrowser.append(self.last_log_html)

    def simple_doocs_read(self, addr):
        """ DOOCS read function """
        if do_doocs:
            try:
                v = pydoocs.read(addr)['data']
            except Exception as err:
                print(addr, err)
                v = 23
        else:
            v = 23
        return v

    def simple_doocs_write(self, addr, value):
        """ DOOCS write function """
        if do_doocs:
            try:
                x = pydoocs.write(addr, value)
                v = 1
            except Exception as err:
                print(addr, err)
                v = 0
        else:
            v = 0
        return v

    def fetch_doocs_data(self):
        """ Get measurement settings/CRL indicators from DOOCS and display to the panel """
        undulators = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/MEASURED_UNDULATORS'))
        if undulators == 1:
            undulator_name = 'SASE1'
        elif undulators == 2:
            undulator_name = 'SASE2'
        elif undulators == 3:
            undulator_name = 'SASE3'
        elif undulators == 12:
            undulator_name = 'SASE1 & SASE2'
        elif undulators == 13:
            undulator_name = 'SASE1 & SASE3'
        elif undulators == 23:
            undulator_name = 'SASE2 & SASE3'
        elif undulators == 123:
            undulator_name = 'SASE1 & SASE2 & SASE3'
        else:
            undulator_name = 'SASE1 & SASE2 & SASE3'
        self.ui.SASEoptions.setCurrentText(undulator_name)

        sa1_k = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA1')*10000)
        self.ui.sa1_k.setValue(sa1_k)
        sa2_k = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA2')*10000)
        self.ui.sa2_k.setValue(sa2_k)
        sa3_k = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA3')*10000)
        self.ui.sa3_k.setValue(sa3_k)
        meas_time = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/N_MEASUREMENTS'))
        self.ui.measurement_time.setValue(meas_time)
        iterations = int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/DAQ/N_ITERATIONS'))
        self.ui.iterations.setValue(iterations)
        self.check_crls()
        self.ui.log.setText('Fetched data from DOOCS')

    def check_crls(self):
        """ Check CRL indicators from DOOCS and change icon color """
        # SASE1 CRL indicators
        sa1_crl1 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS1.OUT1.STATE'))
        sa1_crl2 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS2.OUT1.STATE'))
        sa1_crl3 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS3.OUT1.STATE'))
        sa1_crl4 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS4.OUT1.STATE'))
        sa1_crl5 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS5.OUT1.STATE'))
        sa1_crl6 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS6.OUT1.STATE'))
        sa1_crl7 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS7.OUT1.STATE'))
        sa1_crl8 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS8.OUT1.STATE'))
        sa1_crl9 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS9.OUT1.STATE'))
        sa1_crl10 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA1_XTD2_CRL/LENS10.OUT1.STATE'))
        sa1_crls = [sa1_crl1, sa1_crl2, sa1_crl3, sa1_crl4, sa1_crl5, sa1_crl6, sa1_crl7, sa1_crl8, sa1_crl9, sa1_crl10]
        if 'OFF' in sa1_crls:
            self.ui.warning.setText('Warning: One or more focusing lens (CRL) are inserted in SA1.')
        else:
            self.ui.warning.setText('')
        self.change_crl_icon(sa1_crl1, self.ui.labelStatusFan1_1)
        self.change_crl_icon(sa1_crl2, self.ui.labelStatusFan1_2)
        self.change_crl_icon(sa1_crl3, self.ui.labelStatusFan1_3)
        self.change_crl_icon(sa1_crl4, self.ui.labelStatusFan1_4)
        self.change_crl_icon(sa1_crl5, self.ui.labelStatusFan1_5)
        self.change_crl_icon(sa1_crl6, self.ui.labelStatusFan1_6)
        self.change_crl_icon(sa1_crl7, self.ui.labelStatusFan1_7)
        self.change_crl_icon(sa1_crl8, self.ui.labelStatusFan1_8)
        self.change_crl_icon(sa1_crl9, self.ui.labelStatusFan1_9)
        self.change_crl_icon(sa1_crl10, self.ui.labelStatusFan1_10)

        # SASE 2 CRL indicators
        sa2_crl1 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS1.OUT1.STATE'))
        sa2_crl2 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS2.OUT1.STATE'))
        sa2_crl3 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS3.OUT1.STATE'))
        sa2_crl4 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS4.OUT1.STATE'))
        sa2_crl5 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS5.OUT1.STATE'))
        sa2_crl6 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS6.OUT1.STATE'))
        sa2_crl7 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS7.OUT1.STATE'))
        sa2_crl8 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS8.OUT1.STATE'))
        sa2_crl9 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS9.OUT1.STATE'))
        sa2_crl10 = str(self.simple_doocs_read('XFEL.FEL/CRL.SWITCH/SA2_XTD1_CRL/LENS10.OUT1.STATE'))
        sa2_crls = [sa2_crl1, sa2_crl2, sa2_crl3, sa2_crl4, sa2_crl5, sa2_crl6, sa2_crl7, sa2_crl8, sa2_crl9, sa2_crl10]
        if 'OFF' in sa2_crls:
            self.ui.warning.setText('Warning: One or more focusing lens (CRL) are inserted in SA2.')
        else:
            self.ui.warning.setText('')

        if 'OFF' in sa2_crls and 'OFF' in sa1_crls:
            self.ui.warning.setText('Warning: One or more focusing lens (CRL) are inserted in SA1 and SA2.')

        self.change_crl_icon(sa2_crl1, self.ui.labelStatusFan2_1)
        self.change_crl_icon(sa2_crl2, self.ui.labelStatusFan2_2)
        self.change_crl_icon(sa2_crl3, self.ui.labelStatusFan2_3)
        self.change_crl_icon(sa2_crl4, self.ui.labelStatusFan2_4)
        self.change_crl_icon(sa2_crl5, self.ui.labelStatusFan2_5)
        self.change_crl_icon(sa2_crl6, self.ui.labelStatusFan2_6)
        self.change_crl_icon(sa2_crl7, self.ui.labelStatusFan2_7)
        self.change_crl_icon(sa2_crl8, self.ui.labelStatusFan2_8)
        self.change_crl_icon(sa2_crl9, self.ui.labelStatusFan2_9)
        self.change_crl_icon(sa2_crl10, self.ui.labelStatusFan2_10)

    def change_crl_icon(self, state, crl):
        """ Update the CRL indicator status. """
        if state == 'ON':
            crl.setPixmap(QtGui.QPixmap(ICON_GREEN_LED))
        elif state == 'OFF':
            crl.setPixmap(QtGui.QPixmap(ICON_PURPLE_LED))
        else:
            crl.setPixmap(QtGui.QPixmap(ICON_GREY_LED))


    def write_doocs_data(self):
        """ Write the measurement settings to DOOCS """
        self.set_config('SA1')
        undulators =  ''.join(filter(str.isdigit, self.ui.SASEoptions.currentText()) )
        und_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/MEASURED_UNDULATORS', undulators)
        k_range_sa1 = self.ui.sa1_k.value()/10000
        sa1_k_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA1', k_range_sa1)
        k_range_sa2 = self.ui.sa2_k.value()/10000
        sa2_k_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA2', k_range_sa2)
        k_range_sa3 = self.ui.sa3_k.value()/10000
        sa3_k_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/K_RANGE_SA3', k_range_sa3)
        meas_time_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/N_MEASUREMENTS', self.ui.measurement_time.value())
        iterations_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/DAQ/N_ITERATIONS', self.ui.iterations.value())
        self.ui.log.setText('Wrote data to DOOCS')

    def set_config(self, SASE):
        """ Get measurement time set in the Settings tab and write this number to the DXMAF configuration file  """
        duration = (self.ui.measurement_time.value()/10)*self.ui.iterations.value()
        self.config_file = self.config_path + 'datalog_writer_'+SASE+'.conf'
        with open(self.config_file, 'r') as file:
            cur_yaml = yaml.safe_load(file)
            cur_yaml.update({'duration': str(timedelta(seconds=duration)+timedelta(seconds=30))})
        with open(self.config_file,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump

    def set_config_predictor(self, SASE, date):
        """ Get Model Date and write this number to the DXMAF configuration file  """
        self.config_file = self.config_path + 'datalog_'+SASE+'.conf'
        with open(self.config_file, 'r') as file:
            cur_yaml = yaml.safe_load(file)
            print(cur_yaml)
            cur_yaml['application'][0]['args']['run'] = str(date)
        with open(self.config_file,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump

    def update_estimated_time(self):
        """ Convert the measurement and iteration settings from the Settings tab to an estimated time in minutes """
        time = np.round((self.ui.measurement_time.value()/600)*self.ui.iterations.value(), 2)
        self.ui.total_meas_time.setText(str(time))

    def list_directories(self):
        """ List directories in the model path """
        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(self.data_path)
        self.dirModel.setFilter(QtCore.QDir.AllDirs|QtCore.QDir.NoDotAndDotDot) # only show up to the folder level
        self.modModel = QFileSystemModel()
        self.modModel.setRootPath(self.model_path)

        self.ui.daq.setModel(self.dirModel)
        self.ui.daq.setRootIndex(self.dirModel.index(self.data_path))
        self.ui.daq.setSortingEnabled(True)
        # Make sure first column width is stretched
        self.ui.daq.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.ui.models.setModel(self.modModel)
        self.ui.models.setRootIndex(self.modModel.index(self.model_path))
        self.ui.models.setSortingEnabled(True)
        # Make sure first column width is stretched
        self.ui.models.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.ui.daq.clicked.connect(self.on_clicked_daq)
        self.ui.models.clicked.connect(self.on_clicked_models)

    def on_clicked_daq(self, index):
        """ On clicking the directory in the Model tab, save the path """
        self.ui.trainmodel_button.setEnabled(True)
        self.train_data_path = self.dirModel.fileInfo(index).absoluteFilePath()

    def on_clicked_models(self, index):
        """ On clicking the directory in the Model tab, save the path """
        self.ui.launchmodel_button.setEnabled(True)
        self.launch_model_path = self.dirModel.fileInfo(index).absoluteFilePath()

    def train_model(self):
        """ Switch a flag in DOOCS to start the model training procedure, also writing the date and the status """
        if 'SA1' in self.train_data_path and '-' in self.train_data_path.split('/')[-1] :
            date = self.train_data_path.split('/')[-1]
            self.ui.model_log.setText('Model training for SA1 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_FLAG')) == 0:
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_DATE', date)
                start_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_FLAG', 1)
                status_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_STATUS', 'REQUESTED')
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being trained.')
        elif 'SA2' in self.train_data_path and '-' in self.train_data_path.split('/')[-1] :
            date = self.train_data_path.split('/')[-1]
            self.ui.model_log.setText('Model training for SA2 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_FLAG')) == 0:
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_DATE', date)
                start_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_FLAG', 1)
                status_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_STATUS', 'REQUESTED')
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being trained.')
        elif 'SA3' in self.train_data_path and '-' in self.train_data_path.split('/')[-1] :
            date = self.train_data_path.split('/')[-1]
            self.ui.model_log.setText('Model training for SA3 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_FLAG')) == 0:
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_DATE', date)
                start_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_FLAG', 1)
                status_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_STATUS', 'REQUESTED')
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being trained.')
        else:
            self.ui.model_log.setText('Not a valid data location')

    def launch_model(self):
        """ Switch a flag in DOOCS to launch the model predictions """
        if 'SA1' in self.launch_model_path and '-' in self.launch_model_path.split('/')[-1] :
            date = self.launch_model_path.split('/')[-1].replace('-', '_')
            self.ui.model_log.setText('Model update for SA1 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/UPDATE_MODEL_FLAG')) == 0:
                self.set_config_predictor('SA1', date)
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/CURRENT_MODEL_DATE', date)
                start_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/UPDATE_MODEL_FLAG', 1)
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being updated.')
        elif 'SA2' in self.launch_model_path and '-' in self.launch_model_path.split('/')[-1] :
            date = self.launch_model_path.split('/')[-1].replace('-', '_')
            self.ui.model_log.setText('Model update for SA2 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/UPDATE_MODEL_FLAG')) == 0:
                self.set_config_predictor('SA2', date)
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/CURRENT_MODEL_DATE', date)
                update_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/UPDATE_MODEL_FLAG', 1)
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being updated.')
        elif 'SA3' in self.launch_model_path and '-' in self.launch_model_path.split('/')[-1] :
            date = self.launch_model_path.split('/')[-1].replace('-', '_')
            self.ui.model_log.setText('Model update for SA3 data from ' + date + ' requested.')
            if int(self.simple_doocs_read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/UPDATE_MODEL_FLAG')) == 0:
                self.set_config_predictor('SA3', date)
                date_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/CURRENT_MODEL_DATE', date)
                update_flag = self.simple_doocs_write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/UPDATE_MODEL_FLAG', 1)
            else:
                self.ui.model_log.setText('Error writing data to DOOCS. Model is already being updated.')
        else:
            self.ui.model_log.setText('Not a valid model location')

    def restart_image_analysis_server(self):
        """ Restart image analysis server """
        restart_flag = self.simple_doocs_write('XFEL.UTIL/PY_IMAGE_ANALYSIS/XFELML3._SVR/SVR.STOP_SVR', 1)

    def restart_model_training_server(self):
        """ Restart model training server """
        restart_flag = self.simple_doocs_write('XFEL.UTIL/PY_BEAM_POINTING_TRAINING/XFELML3._SVR/SVR.STOP_SVR', 1)

    def restart_model_prediction_server(self):
        """ Restart model prediction server """
        restart_flag = self.simple_doocs_write('XFEL.UTIL/PY_BEAM_POINTING_PREDICTION/XFELML3._SVR/SVR.STOP_SVR', 1)

    def makedirs(self, dest):
        """ Create a directory if it does not exist """
        if not os.path.exists(dest):
            os.makedirs(dest)

    def deletedirs(self):
        """ Delete temporary folders """
        path = os.getcwd()
        file_path = path + '/temp/'
        try:
            shutil.rmtree(file_path)
            print("Temp file deleted.")
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))

    def error_box(self, message):
        QtGui.QMessageBox.about(self, "Error box", message)

    def question_box(self, message):
        #QtGui.QMessageBox.question(self, "Question box", message)
        reply = QtGui.QMessageBox.question(self, "Question Box",
                                           message,
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            return True

        return False

    def logbook_entry(self, text=""):
        """
        Method to send data + screenshot to eLogbook
        :return:
        """
        #screenshot = self.get_screenshot(widget)
        res = send_to_desy_elog(
            author="xfeloper", title="Beam Pointing DAQ Measurement", severity="INFO", text=text, elog="xfellog")
        if res == True:
            success_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Finished scan! Logbook entry submitted. <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
            self.ui.textBrowser.append(success_log_html)
        if not res:
            error_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Finished scan! Error sending eLogBook entry. <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
            self.ui.textBrowser.append(error_log_html)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DAQApp()

    path = 'gui/xfel.png'
    app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()

    sys.exit(app.exec_())
