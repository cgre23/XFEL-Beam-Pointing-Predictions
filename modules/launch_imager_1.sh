#!/bin/sh

#cd /home/xfeloper/user/chgrech/xfel-daq-ui/modules
cd /home/grechc/Documents/xfel_pubsub
source venv/bin/activate

#cd modules/daq

python -m dxmaf -c docs/image_processor_SA1.conf
deactivate
