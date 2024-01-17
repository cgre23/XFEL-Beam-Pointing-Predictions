#. /net/xfeluser1/export/doocs/server/daq_server/ENVIRONMENT.new;
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$LD_LIBRARY_PATH;
#export PATH=/opt/anaconda/bin:$PATH:/export/doocs/bin;
#export PYTHONPATH=$PYTHONPATH_OLD:/home/doocsadm/bm/python/DAQ/classes:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$PYTHONPATH;
#date=`date +'%Y-%m-%d'`
#date='2023-11-20'
SASE='SA1'

#. /opt/anaconda/etc/profile.d/conda.sh
#conda init bash
#conda deactivate
#conda activate dxmaf
date=`python pydoocs_read.py "$SASE"`
#echo $date
mkdir -p -m 777 "modules/models/$SASE/$date"


python NN_train_dxmaf.py --SASE $SASE --run_name $date --properties "models/$SASE/" --source "daq/runs/$SASE/" --label "$SASE-$date"
#conda deactivate
