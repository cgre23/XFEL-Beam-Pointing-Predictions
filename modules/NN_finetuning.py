import os
import sys
import getopt
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from datetime import datetime
import logging
import yaml
import pydoocs
# Set up logging and suppress warnings
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random_seed = 1

# Downsample factor for data
n = 1

# Define the neural network class
class NN(nn.Module):
    def __init__(self, no_hidden_nodes, no_hidden_layers, INPUTS, OUTPUTS):
        super(NN, self).__init__()
        
        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(INPUTS, no_hidden_nodes))
        layers.append(nn.ReLU())   #nn.ReLU()
        for i in range(no_hidden_layers):
            layers.append(nn.Linear(no_hidden_nodes, no_hidden_nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(no_hidden_nodes, OUTPUTS))
        
        # Create the sequential model with the defined layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# Define a custom Dataset class for the DataLoader
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df[features]
        self.targets = df[targets]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
        return x, y
    
# Function to print help and exit the program
def HelpAndExit():
    logging.error("This program trains a PyTorch model based on daq data. Provide SASE and date")
    logging.error("\t-h\t\t- prints this help\n")
    sys.exit(1)
    
# Function to display a fatal error and exit the program
def Fatal(msg):
    sys.stderr.write("%s: %s\n\n" % (os.path.basename(sys.argv[0]), msg))
    HelpAndExit()
    
# Function to calculate root mean squared error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Function to train the neural network model for an epoch
def train_model(model, epoch, train_loader, optimizer, log_interval):
    train_losses = []
    score = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.mse_loss(output.to(device), target.to(device))
        loss.backward()
        optimizer.step()

        out = output.detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        r2 = pearsonr(targets.flatten(), out.flatten())[0] ** 2
        score.append(r2)

        if batch_idx % log_interval == 0:
            loss_n, current = loss.item(), batch_idx * len(data)
            logging.info(f"epoch: {epoch} loss: {loss_n:.5f} r2 {r2:.5f} [{current:>5d}/{len(train_loader.dataset):>5d}]")

        train_losses.append(loss.item())

    return train_losses, np.mean(score)

# Function to validate the neural network model
def validation_model(model, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data.to(device))
            valid_loss += F.mse_loss(output.to(device), target.to(device), reduction='sum').item()

    valid_loss /= len(valid_loader.dataset)
    return valid_loss


def read_folder(data_folder):
   files = os.listdir(data_folder)
   df = []
   for f in files:
    if f.endswith('parquet.gzip'):
        data_file = data_folder + "/" + f
        logging.info('Reading data from:     %s', data_file)
        data_df = pd.read_parquet(data_file, engine='fastparquet').reset_index(drop=True)
        data_df = data_df.astype(str).replace({"\[":"", "\]":""}, regex=True).astype(float)
        df.append(data_df)
   df_full = pd.concat(df, ignore_index=True)
   return df_full

# Function to parse input parameters
def input_params(argv):
    SASE = None
    run_name = ''
    properties = None
    source = None
    label = ''
    
    try:
        opts, args = getopt.getopt(argv, "hs:t:p:f:l", ["SASE=", "run_name=", "properties=", "source=", "label="])
    except getopt.GetoptError:
        HelpAndExit()
    
    for opt, arg in opts:
        if opt == '-h':
            HelpAndExit()
        elif opt in ("-s", "--SASE"):
            SASE = arg
        elif opt in ("-t", "--run_name"):
            run_name = arg
        elif opt in ("-p", "--properties"):
            properties = arg
        elif opt in ("-f", "--source"):
            source = arg
        elif opt in ("-l", "--label"):
            label = arg 
    
    if not properties:
        properties = './'
    if (not os.path.exists(properties)) or (not os.access(properties, os.W_OK)):
        Fatal("Directory for files '%s' doesn't exist or not writable" % properties)
        
    if not properties.endswith('/'):
        properties += '/'
    
    if not SASE or not run_name:
        Fatal("Please, check your input arguments and make sure to insert at least an undulator number and a run name")
    
    if run_name:
        try:
            run_name_str = str(run_name)
        except:
            Fatal("Please, check run name format '%s'. It must be a string. " % run_name) 
    else:
         Fatal("Provide the date in the run name string")
            
    if label:
        try:
            label_str =  str(label)
        except:
            Fatal("Please, check label format '%s'. It must be a string. " % label) 
    else:
         Fatal("Provide a label for the run in the label string")
        
    return run_name_str, SASE, properties, source, label_str

if __name__ == "__main__":
    run, SASE_no, properties, source, label = input_params(sys.argv[1:])
    
    # Record the start time for training
    start_time = datetime.now()  
    logging.info('Training a NN model for %s data from the folder %s', SASE_no, source+run+'/retrain')
    
    data = {}
    
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    cwd = os.getcwd()
    log_interval = 1000
  
    # Read the data and prepare features and targets
    
    #with open(properties + "channels_" + SASE_no + ".yml", 'r') as file:
    #    channel_list = yaml.load(file, Loader=yaml.Loader)
            
    metadata_file = properties+run+'/metadata_post_training_'+label+'.json'
    filename_pt_retrain = properties+run+'/retrain/metadata_post_training_'+label+'.json'
    try:
        df = read_folder(source+run+'/retrain')

    except FileNotFoundError:
        Fatal('Trouble loading files.')

    try:
        data = json.load(open(metadata_file))
        features = data['features']
        targets = data['targets']
        HIDDEN_NODES = data['hidden_nodes']
        HIDDEN_LAYERS = data['hidden_layers']
        INPUTS = data['no_inputs']
        OUTPUTS = len(targets)
        LEARNING_RATE = data['learning_rate'] / 2
        BATCH_SIZE = data['batch_size']
        MOMENTUM = float(data['momentum'])
    except FileNotFoundError:
        Fatal('Metadata file not found.')

    # Some data preprocessing steps...
    # Remove columns with a zero standard deviation. Otherwise an issue could be caused with normalization
    df=df.loc[:, df.std() > 0]

    inputs_outputs = data['inputs_outputs']
    df = df[inputs_outputs]
    dfmin = pd.Series(index=list(data['norm_min'].keys()), data=data['norm_min'])
    dfmax = pd.Series(index=list(data['norm_max'].keys()), data=data['norm_max'])
    
    # Split the data into training, validation, and testing sets
    traindf, validdf = np.split(df[inputs_outputs], [int(.85*len(df))])
    traindf = traindf.astype(float).dropna()
    
    # Normalize the training and validation data
    norm_df=((traindf-dfmin)/(dfmax-dfmin))
    normvalid_df=((validdf-dfmin)/(dfmax-dfmin))
    norm_df = norm_df.astype(float).dropna().dropna(axis=1, how='all')
    normvalid_df = normvalid_df.astype(float).dropna().dropna(axis=1, how='all')
    logging.info('Number of training samples: %d', len(norm_df))
    
    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if features == [] or targets == []:
        Fatal('Error: Empty features or targets list.')
         
    
    # Get neural network architecture details from the data
    logging.info('Training NN: Learning rate: %s Hidden layers: %s Hidden nodes: %s Batch size: %s', LEARNING_RATE, HIDDEN_LAYERS, HIDDEN_NODES, BATCH_SIZE)

    # Create the neural network model
    model = NN(HIDDEN_NODES, HIDDEN_LAYERS, INPUTS, OUTPUTS)
    try:
        model.load_state_dict(torch.load(properties+run+f'/model-{run}-{label}.pth'))
    except:
        Fatal('Trouble loading model.')
    model = model.to(device)
    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Create DataLoader for training and validation
    train_dataset = MyDataset(norm_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    valid_dataset = MyDataset(normvalid_df)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Early stopping variables
    last_loss = 100
    patience = 4
    trigger_times = 0

    # Training loop
    for epoch in range(300):
        train_loss, mean_train_r2 = train_model(model, epoch + 1, train_loader, optimizer, log_interval)
        current_loss = validation_model(model, valid_loader)

        if current_loss > last_loss and epoch > 5:
            trigger_times += 1
            logging.info('Trigger Times: %d', trigger_times)

            if trigger_times >= patience:
                logging.info('Early stopping!')
                break
        else:
            trigger_times = 0
        last_loss = current_loss
        
    # Save the trained model and optimizer
    if os.path.exists(properties + run + f'/retrain/model-{run}-{label}.pth'):
        os.remove(properties + run + f'/retrain/model-{run}-{label}.pth')        
    torch.save(model.state_dict(), properties + run + f'/retrain/model-{run}-{label}.pth')
    
    if os.path.exists(properties + run + f'/retrain/{run}_optimizer.pth'):
        os.remove(properties + run + f'/retrain/{run}_optimizer.pth')
    torch.save(OPTIMIZER.state_dict(), properties + run + f'/retrain/{run}_optimizer.pth')

    # Training process is complete. Save the results in JSON file.
    logging.info('Retraining process has finished with R2 %.4f. Saving retrained model.', mean_train_r2)
    
    # Evaluate the model on the validation dataset
    score = []
    model.eval()
    output_l = []
    with torch.no_grad():
        for dataset, target in valid_loader:
            output = model(dataset.to(device))
            pred = output.data.max(1, keepdim=True)[1]
            out = output.detach().cpu().numpy()
            output_l.append(out)
            targets = target.detach().cpu().numpy()
            valid_r2 = pearsonr(targets.flatten(), out.flatten())[0] ** 2
            score.append(valid_r2)
    mean_valid_r2 = np.mean(score)
    
    # Serialize data into file:
    stop_time = datetime.now()    
    data['run'] = run
    data['training_start_datetime'] = start_time.strftime("%d/%m/%Y %H:%M:%S")
    data['training_stop_datetime'] = stop_time.strftime("%d/%m/%Y %H:%M:%S")
    data['actual_epochs'] = epoch
    data['early_stopping_patience'] = patience
    data['training_R2'] = mean_train_r2
    data['validation_R2'] = mean_valid_r2
    data['batch_size'] = BATCH_SIZE
    data['learning_rate'] = LEARNING_RATE
    data['momentum'] = MOMENTUM
    data['hidden_layers'] = HIDDEN_LAYERS
    data['hidden_nodes'] = HIDDEN_NODES
    data['label'] = label
    data['start_time'] = str(start_time)
    data['stop_time'] = str(stop_time)
    
    # Serialize data into file:  
    if os.path.exists(filename_pt_retrain):
        os.remove(filename_pt_retrain)
    
    try:
        json.dump(data, open(filename_pt_retrain, 'w'), default=str)
        logging.info('JSON property file saved to: %s', filename_pt_retrain)
    except:
        Fatal('Trouble saving model metadata to file.')
        
    pydoocs.write('XFEL.UTIL/DYNPROP/BEAM_PREDICT.'+SASE_no+'/TRAIN_MODEL_STATUS', 'FINISHED')
    
        
        
 
