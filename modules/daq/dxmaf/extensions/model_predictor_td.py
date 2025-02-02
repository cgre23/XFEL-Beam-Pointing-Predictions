import json
import logging
import os
import pandas as pd
import tempfile
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Set, Any, Optional, Mapping
import numpy
import pydoocs
import torch
import torch.nn as nn
from dxmaf.data_subscriber import BufferedDataSubscriber
from dxmaf.DaqTimingPattern import *
from collections import namedtuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TaggedData = namedtuple('TaggedData', ['sequence_id', 'data'])
doocs_write = 1

# Define the neural network class
class NN(nn.Module):
    def __init__(self, no_hidden_nodes, no_hidden_layers, INPUTS, OUTPUTS):
        super(NN, self).__init__()
        
        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(INPUTS, no_hidden_nodes))
        layers.append(nn.ReLU())
        for i in range(no_hidden_layers):
            layers.append(nn.Linear(no_hidden_nodes, no_hidden_nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(no_hidden_nodes, OUTPUTS))
        
        # Create the sequential model with the defined layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ModelPredictorTD(BufferedDataSubscriber):
    """
    DxMAF module that writes data from subscribed channels to numpy and metadata files.
    """

    def __init__(self, channels: Set[str], SASE: str, model_path: str, run: str, record_data: Optional[bool] = False, output_file: Optional[str] = None,):
        """
        Initializes the ModelPredictor object.

        :param channels:              Set (unique sequence) of DOOCS channel addresses for which `process` will be
                                      called in the event of new data.
        """

        BufferedDataSubscriber.__init__(self, channels, len(channels))
        self.channels = channels
        self.record_data = record_data
        self.SASE = SASE
        self.x=0
        print('Number of channels:', len(channels))
        
        # Load data from JSON file 
        runname = run.replace('_', '-')
        data = json.load( open( model_path+runname+'/metadata_post_training_'+SASE+'-'+str(runname)+'.json' ) )
        f = lambda x: x.replace("/XFEL", "XFEL").replace("/X.TD", "/X."+SASE).replace("/Y.TD", "/Y."+SASE).replace("/Value", "")
        g = lambda x: x.replace("_X_MEASUREMENT", "_X_PREDICTION.TD").replace("_Y_MEASUREMENT", "_Y_PREDICTION.TD")
        
        # Extract features and targets from the data. Transform DOOCS channel names.
        self.features = list(map(f, data['features']))
        self.targets = list(map(g, data['targets'])) 
        
 
        norm_min = {k.replace("/XFEL", "XFEL").replace("/X.TD", "/X."+SASE).replace("/Y.TD", "/Y."+SASE).replace("/Value", "").replace("_X_MEASUREMENT", "_X_PREDICTION.TD").replace("_Y_MEASUREMENT", "_Y_PREDICTION.TD"): v for k, v in data['norm_min'].items()}
        norm_max = {k.replace("/XFEL", "XFEL").replace("/X.TD", "/X."+SASE).replace("/Y.TD", "/Y."+SASE).replace("/Value", "").replace("_X_MEASUREMENT", "_X_PREDICTION.TD").replace("_Y_MEASUREMENT", "_Y_PREDICTION.TD"): v for k, v in data['norm_max'].items()}

        self.dfmin = pd.Series(index=norm_min.keys(), data=norm_min)
        self.dfmax = pd.Series(index=norm_max.keys(), data=norm_max)
        logging.info('Loaded model metadata')
        
        
        if self.record_data == True:
            export_columns = self.targets.copy()
            self.df_export = pd.DataFrame(columns=export_columns)
            self.output_file = Path(datetime.now().strftime(output_file))
        
        # Get neural network architecture details from the data
        HIDDEN_NODES = data['hidden_nodes']
        HIDDEN_LAYERS =data['hidden_layers']
        INPUTS=data['no_inputs']
        OUTPUTS = len(self.targets)
        # Load the pre-trained model
        self.model = NN(HIDDEN_NODES, HIDDEN_LAYERS, INPUTS, OUTPUTS)
        self.model.load_state_dict(torch.load(model_path+f'/{runname}/model-{runname}-{SASE}-{runname}.pth'))
        logging.info('Loading model from %s', model_path+f'/{runname}/model-{runname}-{SASE}-{runname}.pth')
        logging.info('Model loaded: %s', self.model.eval())
        #print(self.dfmin)
        #print(self.dfmax)
       
        
        
    def process_incomplete(self, dataset: Mapping[str, numpy.ndarray], sequence_id: int) -> None:
        #logging.info('Process Incomplete')
        pass
        
    def process_complete(self, dataset: Mapping[str, float], sequence_id: int) -> None:
        val = {}
        padded_val = {}
        a_dic={}
        output_array = []
        length = len(self.filter_indices)

        for idex, iteration in enumerate(self.filter_indices):
            #print(idex, self.filter_indices)
            for k, v in dataset.items():
                if 'BPM' in k and self.filter_indices != []:
                    try:
                        a_dic[k] = v[idex] 
                    except:
                        logging.info('Waiting for bunch number change......')
                        continue
                else:
                    a_dic[k] = v

            df = pd.DataFrame(a_dic, index = [0])
            try:
                df = df[self.features]
                normdf = (df[self.features]-self.dfmin[self.features])/(self.dfmax[self.features]-self.dfmin[self.features])
            except:
                continue
        #logging.info("To Model")
        # Test the model and get the prediction
        
            outp = self.model(torch.tensor(normdf.values.astype(numpy.float32))).detach().numpy()
            output_array.append(outp)
            if idex == 0:
                for idx, target in enumerate(self.targets):
                    val[target] = (outp[:,idx]*(self.dfmax[target]-self.dfmin[target])+self.dfmin[target])
                    if doocs_write == 1:
                	    pydoocs.write(target.replace('.TD', ''), val[target])
                	    logging.info('%s, %.3f', target.replace('.TD', ''), val[target])
               
            #logging.info(output_array)
            if idex == len(self.filter_indices)-1:
                for idx, target in enumerate(self.targets):
                    arr = [element[0][idx] for element in output_array]
                    #print(target, arr[0], self.dfmax[target], self.dfmin[target])
                    val[target] = ((numpy.array(arr)*(self.dfmax[target]-self.dfmin[target]))+self.dfmin[target])
                    pad_width=400-len(self.filter_indices)
                    padded_val[target] = np.pad(val[target], (0, pad_width), 'constant')
                    #print(padded_val[target])
                    #val[target] = (output_array[idx])
                    if doocs_write == 1:
                        pydoocs.write(target, padded_val[target])
                        
                        #pydoocs.write(target.replace('.TD', ''), padded_val[target][0])
                        #logging.info('%s, %.3f', target.replace('.TD', ''), padded_val[target][0])
                    else:
                        logging.info('%s, %.3f', target, val[target])
        #if self.record_data == True:
        #    self.df_export.loc[sequence_id]=val

    def bunch_pattern_filter(self):
        """
        Read the bunch pattern and get the indices by destination for BPMs

        :return:    List of indices for the relevant destination (SA1/SA2/SA3)
        """
        
        pattern = pydoocs.read('XFEL.DIAG/TIMER.CENTRAL/MASTER/BUNCH_PATTERN')['data']
        data = pattern[::2] 
        data = data[:2708]
        data = [decode_destination(unpack_timing_word(d)) for d in data]
        
        if self.SASE == 'SA1':
            indices = list(np.where([dest_[0] == Destination['xfel'].T4D and dest_[1] != SpecialFlags['xfel'].TLD_SOFT_KICK for dest_ in data])[0])
        elif self.SASE == 'SA2':
            indices = list(np.where([dest_[0] == Destination['xfel'].T5D for dest_ in data])[0])
        elif self.SASE == 'SA3':
            indices = list(np.where([dest_[0] == Destination['xfel'].T4D and dest_[1] == SpecialFlags['xfel'].TLD_SOFT_KICK for dest_ in data])[0])
        
        return indices

    def process(self, channel: str, data: int, sequence_id: int, timestamp: float) -> None:
        """
        Process data from a channel previously subscribed to.

        Buffers data received from individual channels until all data for any given sequence number has been received
        from all channels and then calls `process_complete` to process the complete data set. Data older than the
        specified maximum buffer size is discarded.

        :param channel:     DOOCS address of the channel from which `data` was received.
        :param data:        Read-only data sample from the previously subscribed to channel specified in `channel`.
        :param sequence_id: Sequence ID (macropulse number) of the data sample.
        :param timestamp:   Timestamp of the data sample.
        :return:            None
        """
        

        
        if 'XGM/XGM' in channel:
                if len(data) > 0:
                        data = data[0][1]
                else:
                        data = 0
                        
        # Filter by BUNCHPATTERN destination                
        if 'BPM' in channel:
            channel = channel.replace("/X.TD", "/X."+self.SASE).replace("/Y.TD", "/Y."+self.SASE)
            if self.x == 0 or self.x%100:
                self.filter_indices = self.bunch_pattern_filter()
            data = numpy.take(data[:,1], self.filter_indices)
            
        self.x=self.x+1
        
        highest_sequence_id = max([*self.buffer.keys(), sequence_id])
        if sequence_id < (highest_sequence_id - self.max_buffer_size):
            return

        for key in list(self.buffer.keys()):
            if key < (highest_sequence_id - self.max_buffer_size):
                self.process_incomplete(self.buffer[key], key)
                del self.buffer[key]

        self.buffer.setdefault(sequence_id, {})[channel] = data

        if len(self.buffer[sequence_id]) == len(self.channels):
            self.process_complete(self.buffer[sequence_id], sequence_id)
            del self.buffer[sequence_id]
       

    def close(self) -> None:
        """
        Save data to file when finished or session is interrupted.

        """
        if self.record_data == True:
            logging.info('Writing to file....')
            try:
            	self.df_export.to_parquet(self.output_file, compression='gzip')
            	logging.info('Data saved successfully to: %s', self.output_file)
            except:
            	logging.info('Failed to save data to: %s', self.output_file)
        



# Export DxMAF modules
DXMAF_MODULES = (ModelPredictorTD,)
