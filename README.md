# XFEL-DAQ-UI


## Getting started

This is a user-interface for the Taskomat sequence recording electron diagnostics using DOOCS DAQ and photon diagnostics using Karabo DAQ.

## Add your contribution in the BKR

```
cd xfeloper/user/chgrech/xfel-daq-ui
git pull origin main
git add .
git commit -m 'Message here what changes you are committing'
git push -uf origin main
```

## Description
Change the measurement time, undulators measured and k-ranges. The tool can start the Taskomat measurement sequence as well, and log the result in the logbook at the end.


## Usage
```
python daq.py
```

## Updating the User Interface using Designer

Open the file UIDAQ.ui in the gui folder using DesignerQt to change the layout of the interface. Run these commands to update the interface:
        
        pyrcc5 gui/resources.qrc -o gui/resources_rc.py
        pyuic5 --import-from .  gui/UIDAQ.ui -o gui/UIDAQ.py

## Authors and acknowledgment
Christian Grech, Farzad Jafarinia (MXL).


## Project status
Project is in development
