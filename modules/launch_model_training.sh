#. /net/xfeluser1/export/doocs/server/daq_server/ENVIRONMENT.new;
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$LD_LIBRARY_PATH;
#export PATH=/opt/anaconda/bin:$PATH:/export/doocs/bin;
#export PYTHONPATH=$PYTHONPATH_OLD:/home/doocsadm/bm/python/DAQ/classes:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$PYTHONPATH;
date=`date +'%Y-%m-%d'`
date='2023-11-20'
SASE='SA1'

mkdir -p -m 777 "modules/models/$SASE/$date"
#export PATH=/opt/mambaforge/bin:$PATH
#source activate xfel
#conda deactivate
#conda activate dxmaf
python modules/NN_train_dxmaf.py --SASE $SASE --run_name $date --properties "modules/models/$SASE/" --source "modules/daq/runs/$SASE/" --label "$SASE-$date"
#conda deactivate
