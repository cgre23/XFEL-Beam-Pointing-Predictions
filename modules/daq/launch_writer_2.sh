. /net/xfeluser1/export/doocs/server/daq_server/ENVIRONMENT.new;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$LD_LIBRARY_PATH;
export PATH=/opt/anaconda/bin:$PATH:/export/doocs/bin;
export PYTHONPATH=$PYTHONPATH_OLD:/home/doocsadm/bm/python/DAQ/classes:/export/doocs/lib:/net/doocsdev16/export/doocs/lib:$PYTHONPATH;
#export PATH=/opt/mambaforge/bin:$PATH
#source activate xfel
conda deactivate
conda activate dxmaf
python -m dxmaf -c docs/datalog_writer_SA2.conf
conda deactivate
