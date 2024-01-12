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
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from gui.UIdaq import Ui_Form
import threading, queue
import shutil
import re
from datetime import datetime, timedelta
from modules.spectr_gui import send_to_desy_elog
import gui.resources_rc
import subprocess
import time
do_doocs = 0
if do_doocs == 1:
    import pydoocs
import yaml

# Define status icons (available in the resource file built with "pyrcc5"
ICON_RED_LED = ":/icons/led-red-on.png"
ICON_GREEN_LED = ":/icons/green-led-on.png"
ICON_BLUE_LED = ":/icons/blue-led-on.png"

class DAQApp(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        # Initialize parameters
        self.daterange = 0
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.logstring = []
        self.sa1_sequence_prefix = 'XFEL.UTIL/TASKOMAT/DAQ_SA1'
        self.ui.sequence_button.setCheckable(True)
        self.ui.sequence_button.setEnabled(True)
        self.ui.sequence_button.clicked.connect(self.toggleSequenceButton)
        self.ui.fetch_button.clicked.connect(self.fetch_doocs_data)
        self.ui.write_button.clicked.connect(self.write_doocs_data)
        self.ui.measurement_time.valueChanged.connect(self.update_estimated_time)
        self.ui.iterations.valueChanged.connect(self.update_estimated_time)
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
        self.proc = subprocess.Popen(["/bin/sh",  "./modules/daq/launch_writer_1.sh"], preexec_fn=os.setsid)
    
    def stop_dxmaf(self):
        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
    
    def start_sequence(self):
        try:
           
            pydoocs.write(self.sa1_sequence_prefix+'/RUN.ONCE', 1)
            self.logstring = []
            start_log = datetime.now().isoformat(' ', 'seconds')+': Started Taskomat sequence.\n'
            start_log_html = '<html> <style> p { margin:0px; } span.d { font-size:80%; color:#555555; } span.e { font-weight:bold; color:#FF0000; } span.w { color:#CCAA00; } </style> <body style="font:normal 10px Arial,monospaced; margin:0; padding:0;"> Started the Taskomat sequence.  <span class="d">(datetime)</span></body></html>'.replace('datetime', datetime.now().isoformat(' ', 'seconds'))
            self.logstring.append(start_log)
            self.ui.textBrowser.append(start_log_html)
            while pydoocs.read(self.sa1_sequence_prefix+'/RUNNING')['data'] == 1:
            
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
        self.last_log_html = pydoocs.read(self.sa1_sequence_prefix+'/LOG_HTML.LAST')['data']
        self.last_log = pydoocs.read(self.sa1_sequence_prefix+'/LOG.LAST')['data']
        self.logstring.append(self.last_log+'\n')
        self.ui.textBrowser.append(self.last_log_html)

    def simple_doocs_read(self, addr):
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
            crl.setPixmap(QtGui.QPixmap(ICON_RED_LED))
        else:
            crl.setPixmap(QtGui.QPixmap(ICON_BLUE_LED))


    def write_doocs_data(self):
        self.set_config()
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

    def set_config(self):
        config_file = "modules/daq/docs/datalog_writer_SA1 copy.conf"
        duration = (self.ui.measurement_time.value()/10)*self.ui.iterations.value()
        with open(config_file, 'r') as file:
            cur_yaml = yaml.safe_load(file)
            cur_yaml.update({'duration': str(timedelta(seconds=duration))})
            print(cur_yaml)

        with open(config_file,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump



    def update_estimated_time(self):
        time = np.round((self.ui.measurement_time.value()/600)*self.ui.iterations.value(), 2)
        self.ui.total_meas_time.setText(str(time))

    def makedirs(self, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)

    def deletedirs(self):
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
