import sys
import pydoocs

if sys.argv[1]=='SA1':
    pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_STATUS', 'FINISHED')
    #print('2023-11-27')
if sys.argv[1]=='SA2':
    pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_STATUS', 'FINISHED')
if sys.argv[1]=='SA3':
    pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_STATUS', 'FINISHED')
sys.exit(0)