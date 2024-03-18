import sys
import pydoocs

if sys.argv[1]=='SA1':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/CURRENT_MODEL_DATE')['data'])
    #pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/TRAIN_MODEL_STATUS', 'TRAINING')
    #print('2023-11-27')
if sys.argv[1]=='SA2':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/CURRENT_MODEL_DATE')['data'])
    #pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/TRAIN_MODEL_STATUS', 'TRAINING')
if sys.argv[1]=='SA3':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/CURRENT_MODEL_DATE')['data'])
    #pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/TRAIN_MODEL_STATUS', 'TRAINING')
sys.exit(0)