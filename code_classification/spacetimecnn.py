import os
import time
import torch
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pdb
import random
from collections import OrderedDict
from pprint import pprint
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.modules.utils import _pair
from eeg_new_utils import get_nrmse_torch, unnormalize_envelope
random_state = random_seed = 99 


class Dataset(Dataset):

        def __init__(self, input_paths, target_paths, transform=None):

                self.input_paths = input_paths
                self.target_paths = target_paths
                self.transform = transform

        def __len__(self): # no. of samples     

                return min(len(self.input_paths),len(self.target_paths))

        def __getitem__(self, index): # as ndarray
                X = np.load(self.input_paths[index]).astype(np.float32) 
                y = np.load(self.target_paths[index]).astype(np.float32)

#                if self.transform: # not a transform; just a placeholder b/c lazy loading
 #                       X = torch.tensor(np.array(X, dtype=np.float32).reshape(128,6,5))
  #                      y = torch.tensor(np.array(y, dtype=np.float32)).squeeze(-1)
                return X, y





class SpaceTimeCNN(nn.Module):
        def __init__(self,dropout=0.5):
                super(SpaceTimeCNN, self).__init__()

                self.dropout1 = nn.Dropout(p=dropout)
                self.spatial_cnn = nn.Conv1d(1,25,14)#.double()
                self.nonlin1 = nn.ELU()
                self.dropout2 = nn.Dropout(p=dropout)#.double()
                self.temporal_cnn1= nn.Conv1d(25,25,3)#.double()
                self.nonlin2 = nn.ELU()
                self.dropout3 = nn.Dropout(p=dropout)#.double()
                self.maxpool1 = nn.MaxPool2d((3,1),stride=1)#.double()
                self.temporal_cnn2 = nn.Conv2d(1,25,(3,1))#.double()
                self.nonlin3 = nn.ELU()
                self.maxpool2 = nn.MaxPool2d((3,1),stride=1)#.double()
                self.fully_connected = nn.Linear(120650,256)#.double()


        def forward(self, x):
                
                x = self.dropout1(x)
                x = torch.split(x, 1, dim=1)
                x = torch.stack(x)
                y = []

                                
                for i in range(len(x)):
                        y.append(self.spatial_cnn(x[i]))
                
                
                y = torch.cat(y,2)
                y = y
                y = self.nonlin1(y)
                
                
                y = self.dropout2(y)
                y = self.temporal_cnn1(y)
                y = self.nonlin2(y)
                y = self.maxpool1(y)
                
                y = y.unsqueeze(1)
                y = self.dropout3(y)
                y = self.temporal_cnn2(y)
                y = self.nonlin3(y)
                y = self.maxpool2(y)

                
                #import pdb; pdb.set_trace()
                y = y.view(y.shape[0],-1)
                #import pdb; pdb.set_trace()
                y = self.fully_connected(y) #.unsqueeze(-1)
                
        #       print(y.shape)
#               import pdb; pdb.set_trace()
                
                return(y)





def make_optimizer(net, learning_rate=0.001):
    
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
        return optimizer

def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=100):
        
        train_sampler = RandomSampler(train_dataset)              
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        
        return train_loader, val_loader, test_loader         


def train_network(net, train_loader, val_loader, n_epochs=30, learning_rate=0.065):
        
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("epochs=", n_epochs)
        print("learning_rate=", learning_rate)
        print("=" * 30)
        
        #Get training data
        n_batches = len(train_loader)
        
        #Create our loss and optimizer functions
        optimizer = make_optimizer(net, learning_rate)
        loss = nn.MSELoss()     

        #Time for printing
        training_start_time = time.time()
        
        #Loop for n_epochs
        for epoch in range(n_epochs):
                
                running_loss = 0.0
                print_every = n_batches // 10
                start_time = time.time()
                total_train_loss = 0

                tl = iter(train_loader)
                
                for i in range(len(tl)):

                        data = next(tl)
                        
                        #Get inputs
                        inputs, labels = data
                
                        #Set the parameter gradients to zero
                        optimizer.zero_grad()
                        
                        #Forward pass, backward pass, optimize
                        outputs = net(inputs)
                                                
                        loss_size = loss(outputs, labels) #If you're getting label size errors, this may be the source
                        loss_size.backward()
                        optimizer.step()
                        
                        #Print statistics
                        running_loss += loss_size.data.item()
                        total_train_loss += loss_size.data.item()
                        
                        #Print every 10th batch of an epoch
                        if (i + 1) % (print_every + 1) == 0:
                                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                                                epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                                #Reset running loss and time
                                running_loss = 0.0
                                start_time = time.time()
                        
                #At the end of the epoch, do a pass on the validation set
                total_val_loss = 0
                for inputs, labels in val_loader:
                        
                        #Wrap tensors in Variables
                        #inputs, labels = Variable(inputs), Variable(labels)    
                        
                        #Forward pass
                        val_outputs = net(inputs)
                        val_loss_size = loss(val_outputs, labels)
                        total_val_loss += val_loss_size.data.item()
                        
                print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
        train_time = time.time() - training_start_time

        print("Training finished, took {:.2f}s".format(train_time))

        return(total_val_loss, train_time)

def get_data(input_path, target_path, mode):

        input_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)])
        target_files = sorted([os.path.join(target_path, f) for f in os.listdir(input_path)]) #input path twice is deliberate

        if len(input_files) != len(target_files):
                raise Error
        
        test_size=0.1
        
        train_input, test_input, train_target, test_target = train_test_split(
                input_files, target_files, test_size=test_size, random_state=random_state)

        train_size = 0.9

        train_input, val_input, train_target, val_target = train_test_split(
                train_input, train_target, train_size=train_size, random_state=random_state)

        train_dataset = Dataset(train_input, train_target)
        val_dataset = Dataset(val_input, val_target)
        test_dataset = Dataset(test_input, test_target) 
        

        return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
        
        from utils import root_dir
        import datetime
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=0.065)  
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--sd",action="store_true") 

        args = parser.parse_args()
        
        random_state = random_seed = 99 
        
        
        n_epochs=args.epochs
        learning_rate=args.lr
        dropout = args.dropout
        if args.sd == True:
                mode = "dep"
        else:
                mode = "ind"

        
        input_path = os.path.join(root_dir, "single_bandpass", "windows", "EEG")
        
        target_path = os.path.join(root_dir, "single_bandpass","windows","Envelopes")   
        train_dataset, val_dataset, test_dataset = get_data(input_path, target_path, mode=mode) 
        
        train_loader, val_loader, test_loader = get_loaders(train_dataset, val_dataset, test_dataset, batch_size=100)
        

        net = SpaceTimeCNN(dropout=dropout)
        
        
        net.train()     
        
        val_loss, train_time = train_network(net, train_loader, val_loader, n_epochs=n_epochs, learning_rate=learning_rate)

        nrmse = get_nrmse_torch(val_loader, net)

        net.eval()

        X_true, y_true = next(iter(val_loader)) 
        
        
        y_pred = net(X_true)
        
        
        y_pred = unnormalize_envelope(y_pred.detach().numpy().reshape(-1))
        y_true = unnormalize_envelope(y_true.detach().numpy().reshape(-1))


        results_dict = {"model": "spacetimecnn","epochs": n_epochs,
                 "lr": learning_rate, "val": val_loss, "nrmse": nrmse,
                        "time": train_time, "dropout": dropout}
        
        now = datetime.datetime.now()   

        right_now = now.strftime("%m%d%H:%M")

        results_dir = os.path.join(root_dir, "results")

        save_name = "{0}_{1}_{2}_".format(mode, "space", right_now)
        
        score_name = os.path.join(results_dir, save_name + "score.npy")
        y_true_name = os.path.join(results_dir, save_name + "true.npy") 
        y_pred_name = os.path.join(results_dir, save_name + "pred.npy")

        np.save(score_name, np.array([results_dict]))
        np.save(y_true_name, y_true)
        np.save(y_pred_name, y_pred)


