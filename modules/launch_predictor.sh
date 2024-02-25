#!/bin/sh
SASE='SA1'

cd /home/xfeloper/user/chgrech/xfel-daq-ui/modules
source venv/bin/activate
#export PYTHONPATH=$PYTHONPATH_OLD:/home/xfeloper.nfs/user/chgrech/xfel-daq-ui/modules/venv/lib/python3.9:/local/lib:/home/xfeloper/released_software/python/hlc_toolbox_common:/home/xfeloper/released_software/python/lib:$PYTHONPATH;

cd daq
python -m dxmaf -c docs/datalog_SA1.conf
conda deactivate
