import os
import sys
import getopt
import json
from scipy.stats import pearsonr
import pandas as pd
import warnings
import logging
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
def train_model(model, epoch, train_loader, OPTIMIZER):
    train_losses = []
    train_counter = []
    score = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the network to training mode
    model.train()
    size = len(train_loader) * BATCH_SIZE

    for batch_idx, (data, target) in enumerate(train_loader):
        tot_score = 0
        # Reset gradients
        OPTIMIZER.zero_grad()

        # Evaluate network with data
        output = model(data.to(device))
        
        # Compute loss and derivative
        loss = F.mse_loss(output.to(device), target.to(device))
        loss.backward()

        # Step the optimizer
        OPTIMIZER.step()
        
        out = output.detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        r2 = pearsonr(targets.flatten(), out.flatten())[0] ** 2
        score.append(r2)
        
        # Print out results and save to file
        if batch_idx % log_interval == 0:
            loss_n, current = loss.item(), batch_idx * len(data)
            logging.info(f"epoch: {epoch} loss: {loss_n:>5f} r2 {r2:>3f} [{current:>5d}/{size:>5d}]")
        
        tot_score = np.mean(score)
        train_losses.append(loss.item())
        
    return train_losses, tot_score, model, OPTIMIZER

# Function to validate the neural network model
def validation_model(model, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data.to(device))
            valid_loss += F.mse_loss(output.to(device), target.to(device), reduction='sum').item()
    valid_loss /= len(valid_loader.dataset)
   
    return valid_loss

# Function to reset model weights to avoid weight leakage
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

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
    logging.info('Training a NN model for %s data from the folder %s', SASE_no, source+run)
    
    data = {}
    
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    cwd = os.getcwd()
    log_interval = 1000
  
    # Read the data and prepare features and targets
    df = read_folder(source+run)
    
    with open(properties + "channels_" + SASE_no + ".yml", 'r') as file:
        channel_list = yaml.load(file, Loader=yaml.Loader)
        
    #print(channel_list)
    
    filename_pt = properties+run+'/metadata_post_training_'+label+'.json'
    # Some data preprocessing steps...
    # Remove columns with a zero standard deviation. Otherwise an issue could be caused with normalization
    df=df.loc[:, df.std() > 0]

    # Define the features and targets from the channel list, pre-normalization
    features_pre = list(set(channel_list['inputs']).intersection(df.columns.tolist()))
    targets_pre = list(set(channel_list['outputs']).intersection(df.columns.tolist()))
    #logging.info('Training with %d percent of the data', label*100)
    inputs_outputs = features_pre + targets_pre
    
    # Split the data into training, validation, and testing sets
    traindf, validdf, testdf = np.split(df[inputs_outputs], [int(.75*len(df)), int(.85*len(df))])
    traindf = traindf.astype(float).dropna()
    
    # Calculate normalization min and max values for the data
    min_values = (traindf.min() - traindf.std()).tolist()
    max_values = (traindf.max() + traindf.std()).tolist()
    
    # Convert min and max values to dictionaries
    min_dict = {traindf.columns[i]: min_values[i] for i in range(len(traindf.columns.tolist()))}
    max_dict = {traindf.columns[i]: max_values[i] for i in range(len(traindf.columns.tolist()))}
    data['norm_min'] = min_dict
    data['norm_max'] = max_dict

    # Normalize the training and validation data
    norm_df = (traindf - (traindf.min() - traindf.std())) / ((traindf.max() + traindf.std()) - (traindf.min() - traindf.std()))
    normvalid_df = (validdf - (traindf.min() - traindf.std())) / ((traindf.max() + traindf.std()) - (traindf.min() - traindf.std()))
    norm_df = norm_df.astype(float).dropna().dropna(axis=1, how='all')
    normvalid_df = normvalid_df.astype(float).dropna().dropna(axis=1, how='all')
    logging.info('Number of training samples: %d', len(norm_df))
    
    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the features and targets from the channel list after normalization, some columns may be NaN or empty so they have to be dropped
    features = list(set(norm_df.columns.tolist()).intersection(features_pre))
    targets = list(set(norm_df.columns.tolist()).intersection(targets_pre))
 
    if features == [] or targets == []:
        Fatal('Error: Empty features or targets list.')
        
    #logging.info('Training with %d percent of the data', label*100)
    inputs_outputs = features + targets
    INPUTS = len(features)
    OUTPUTS = len(targets)
    data['no_inputs'] = INPUTS
    data['features'] = features
    data['targets'] = targets
    data['inputs_outputs'] = inputs_outputs
    
    # Set hyperparameters for training
    LEARNING_RATE = channel_list['learning_rate']
    HIDDEN_LAYERS = channel_list['hidden_layers']
    HIDDEN_NODES = channel_list['hidden_nodes']
    BATCH_SIZE = channel_list['batch_size']
    MOMENTUM = float(channel_list['momentum'])
    logging.info('Training NN: Learning rate: %s Hidden layers: %s Hidden nodes: %s Batch size: %s', LEARNING_RATE, HIDDEN_LAYERS, HIDDEN_NODES, BATCH_SIZE)

    # Create the neural network model
    model = NN(HIDDEN_NODES, HIDDEN_LAYERS, INPUTS, OUTPUTS)
    model = model.to(device)
    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Create DataLoader for training and validation
    train_dataset = MyDataset(norm_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    valid_dataset = MyDataset(normvalid_df)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Early stopping variables
    last_loss = 100
    patience = 5
    trigger_times = 0

    # Training loop
    for t in range(1000):
        epoch = t + 1
        train_loss, mean_train_r2, model, OPTIMIZER = train_model(model, epoch, train_loader, OPTIMIZER)
        current_loss = validation_model(model, valid_loader)

        if current_loss > last_loss and epoch > 30:
            trigger_times += 1
            logging.info('Trigger Times: %d', trigger_times)

            if trigger_times >= patience:
                logging.info('Early stopping!')
                break
        else:
            trigger_times = 0
        last_loss = current_loss
        
    # Save the trained model and optimizer
    if os.path.exists(properties + run + f'/model-{run}-{label}.pth'):
        os.remove(properties + run + f'/model-{run}-{label}.pth')        
    torch.save(model.state_dict(), properties + run + f'/model-{run}-{label}.pth')
    
    if os.path.exists(properties + run + f'/{run}_optimizer.pth'):
        os.remove(properties + run + f'/{run}_optimizer.pth')
    torch.save(OPTIMIZER.state_dict(), properties + run + f'/{run}_optimizer.pth')

    # Training process is complete. Save the results in JSON file.
    logging.info('Training process has finished with R2 %.4f. Saving trained model.', mean_train_r2)
    
    # Evaluate the model on the validation dataset
    test_losses, score = [], []
    total_score = 0
    model.eval()
    test_loss, correct = 0, 0
    output_l = []
    with torch.no_grad():
        for dataset, target in valid_loader:
            output = model(dataset.to(device))
            test_loss += F.mse_loss(output.to(device), target.to(device), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            out = output.detach().cpu().numpy()
            output_l.append(out)
            targets = target.detach().cpu().numpy()
            valid_r2 = pearsonr(targets.flatten(), out.flatten())[0] ** 2
            score.append(valid_r2)
    mean_valid_r2 = np.mean(score)
    logging.info('Evaluation with validation dataset: %.4f', mean_valid_r2)
    
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
    data['hidden_layers'] = HIDDEN_LAYERS
    data['hidden_nodes'] = HIDDEN_NODES
    data['label'] = label
    data['start_time'] = str(start_time)
    data['stop_time'] = str(stop_time)
    
    # Serialize data into file:  
    if os.path.exists(filename_pt):
        os.remove(filename_pt)
    
    try:
        json.dump(data, open(filename_pt, 'w'), default=str)
        logging.info('JSON property file saved to: %s', filename_pt)
    except:
        Fatal('Trouble saving model metadata to file.')
        
        
 
