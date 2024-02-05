import sys
import pydoocs

if sys.argv[1]=='SA1':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA1/UPDATE_MODEL_DATE')['data'])
    #print('2023-11-27')
if sys.argv[1]=='SA2':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA2/UPDATE_MODEL_DATE')['data'])
if sys.argv[1]=='SA3':
    print(pydoocs.read('XFEL.UTIL/DYNPROP/BEAM_PREDICT.SA3/UPDATE_MODEL_DATE')['data'])
sys.exit(0)
