import os
import numpy as np
from timeit import timeit
from scipy import integrate, stats
import re
import entropy
#from entropy.entropy import spectral_entropy
import argparse
from utils import root_dir


fs = 128


def mean(x):
        return np.mean(x)

def absmean(x):
        return np.mean(np.abs(x))

def maximum(x):
        return np.max(x)

def absmax(x):
        return np.max(np.abs(x))

def minimum(x):
        return np.min(x)

def absmin(x):
        return np.min(np.abs(x)) 

def minplusmax(x):
        return np.max(x) + np.min(x)

def maxminusmin(x):
        return np.max(x) - np.min(x)

def curvelength(x):
        cl = 0
        for i in range(x.shape[0]-1):
                cl += abs(x[i]-x[i+1])
        return cl

def energy(x):
        return np.sum(np.multiply(x,x))


def nonlinear_energy(x):
        # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
        x_squared = x[1:-1]**2
        subtrahend = x[2:]*x[:-2]
        return np.sum(x_squared-subtrahend)

#def ehf(x,prev):
        #(based on Basar et. al. 1983)
#       "prev" is array of values from prior context
#       rms = np.sqrt(np.mean(prev**2))
#       return 2*np.sqrt(2)*(max(x)/rms)
        

def spec_entropy(x):
        return entropy.spectral_entropy(x,fs,method="welch",normalize=True)

def integral(x):
        return integrate.simps(x)

def stddeviation(x):
        return np.std(x)

def variance(x):
        return np.var(x)

def skew(x):
        return stats.skew(x)

def kurtosis(x):
        return stats.kurtosis(x)

#added ones

#some of these are nicked from https://github.com/raphaelvallat/entropy

def sample_entropy(x):
        return entropy.sample_entropy(x, order=2, metric='chebyshev')  

def perm_entropy(x):
        return entropy.perm_entropy(x, order=3, normalize=True)

def svd_entropy(x):
        return entropy.svd_entropy(x, order=3, delay=1, normalize=True)

def app_entropy(x):
        return entropy.app_entropy(x, order=2, metric='chebyshev')

def petrosian(x):
        return entropy.petrosian_fd(x)

def katz(x):
        return entropy.katz_fd(x)

def higuchi(x):
        return entropy.higuchi_fd(x, kmax=10)

def rootmeansquare(x):
        return np.sqrt(np.mean(x**2))

def dfa(x):
        return entropy.detrended_fluctuation(x)
        

#doesn't contain EHF since that must be added later
#funclist = [mean,absmean,maximum,absmax,minimum,absmin,minplusmax,maxminusmin,curvelength,energy,nonlinear_energy, integral,stddeviation,variance,skew,kurtosis,np.sum,spec_entropy,
#sample_entropy, perm_entropy, svd_entropy, app_entropy,
#petrosian, katz, higuchi, rootmeansquare, dfa] 

funclist = [katz, minplusmax, spec_entropy, variance, minimum, integral, np.sum, mean, nonlinear_energy, rootmeansquare]


def window_data(data:np.ndarray, w_len):
        """windows the data
        (using a stride length of 1)
        """     

        data_len = len(data)

        windowed_data = []
        
        for i in range(w_len):
                to_split = data[i:]
                l_to_crop = len(to_split) % w_len
                final_length = len(to_split) - l_to_crop
                cropped = to_split[:final_length]
                some_split = np.split(cropped,int(len(cropped)/w_len))
                windowed_data.append(some_split)
        
        ordered_windows = []

        for i in range(max([len(w) for w in windowed_data])):
                for j in range(len(windowed_data)):
                        try:
                                ordered_windows.append(windowed_data[j][i])
                        except IndexError:
                                pass
        
        ordered_windows = np.array(ordered_windows)

        print("ordered_windows",ordered_windows.shape)          
        return ordered_windows
        

def feats_array_4_window(window:np.ndarray):    
        """Takes a single window, returns an array of features of 
        shape (n.features, electrodes), and then flattens it 
        into a vector
        """
        print("window shape",window.shape)      

        if len(window.shape) != 2:
                try:
                        window = window.reshape((window.shape[0],1))
                except IndexError:
                        import pdb; pdb.set_trace()
        
        outvec = np.zeros((len(funclist), window.shape[1]))
                
        for i in range(len(funclist)):
                for j in range(window.shape[1]):
                        outvec[i,j] = funclist[i](window[:,j])
        
        print(outvec.shape)
        outvec = outvec.reshape(-1)

        return outvec


def make_simple_feats(windowed_data:np.ndarray):
        
        print(windowed_data.shape)
        
        simple_feats = []
        
        for w in range(len(windowed_data)):
                simple_feats.append(feats_array_4_window(windowed_data[w]))

        return(np.array(simple_feats))


def add_deltas(feats_array:np.ndarray):

        deltas = np.diff(feats_array,axis=0)
        double_deltas = np.diff(deltas,axis=0)
        all_feats = np.hstack((feats_array[2:],deltas[1:],double_deltas))

        return(all_feats)


class FeatureMaker():
        def __init__(self, window_len, delta_feats=False, indices=None):
                self.indices = indices
                self.window_len = window_len
                self.delta_feats = delta_feats

        def set_raw_paths(self, raw_paths_list):
                self.raw_paths = raw_paths_list

        def set_out_dir(self, out_dir):
                self.out_dir = out_dir

        def set_load_function(self, fn): #function that takes the path of some data and returns it
                self.data_fn = fn       

        def make_feats(self):
                for i, d in enumerate(self.raw_paths):
                        data = self.data_fn(d)
                        print(data.shape)
                        if self.indices:
                                data = data[:,self.indices]
                        data = window_data(data, self.window_len)
                        out_feats = make_simple_feats(data)
                        if self.delta_feats:
                                out_feats = add_deltas(out_feats)
                        speaker_number = self.naming_function(d)
                        np.save(os.path.join(self.out_dir, "svm_feats_" + speaker_number.zfill(2)), out_feats)
        
        def set_naming_function(self, nf):
                self.naming_function = nf    #requires a function that takes in a filepath (for EEG data) and 
                                                #returns a speaker id number

def get_speaker_id(filepath:str):
        
        filepath_re = re.compile(r".*\/feis\/experiments\/([0-9]+)\/([a-z]+).npy")
        file_match = re.match(filepath_re, filepath)
        speaker_number = int(file_match[1])
        if file_match[2] == "thinking":
                speaker_number += 21
        return(str(speaker_number))
                                

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("--nobroca", action="store_true")
        args = parser.parse_args()

        if args.nobroca:
            feats_dir = os.path.join(root,"svm_features_all")
        else:
            feats_dir = os.path.join(root,"svm_features")
        raw_dir = os.path.join(root,"experiments")

        if not os.path.exists(feats_dir):
            os.mkdir(feats_dir)

        if args.nobroca:
            feat_maker = FeatureMaker(5, delta_feats=True)
        else:
            feat_maker = FeatureMaker(5, delta_feats=True, indices=1)
        feat_maker.set_out_dir(feats_dir)
        feat_maker.set_load_function(lambda x: np.vstack(np.load(x)))
        feat_maker.set_naming_function(get_speaker_id)

        raw_paths = []
        for s in range(1,22):
                raw_paths.append(os.path.join(raw_dir,str(s).zfill(2),"stimuli.npy"))
                raw_paths.append(os.path.join(raw_dir,str(s).zfill(2),"thinking.npy"))
                
        feat_maker.set_raw_paths(raw_paths)

        feat_maker.make_feats()

