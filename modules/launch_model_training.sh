 #!/bin/sh

SASE='SA1'


cd /home/xfeloper/user/chgrech/xfel-daq-ui/modules
source venv/bin/activate
export PYTHONPATH=/home/xfeloper.nfs/user/chgrech/xfel-daq-ui/modules/venv/lib/python3.9/site-package:$PYTHONPATH

date=`python pydoocs_read.py "$SASE"`
mkdir -p -m777 "models/$SASE/$date"
python3 NN_train_dxmaf.py --SASE $SASE --run_name $date --properties "models/$SASE/" --source "daq/runs/$SASE/" --label "$SASE-$date"
deactivate
