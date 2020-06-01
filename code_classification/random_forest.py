#youwhatmate
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import string
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
#from make_features import make_simple_feats
from utils import root_dir
from eeg_new_utils import get_env_dict, get_nrmse, unnormalize_envelope, get_proper_values
import datetime
from regression_svm import RegressorGridSearch, match_function
import argparse


env_dir = os.path.join(root_dir, "combined_envelopes")
eeg_dir = os.path.join(root_dir, "svm_features")
results_dir = os.path.join(root_dir,"results")


if __name__ == "__main__":      
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", action="store_true")
        parser.add_argument("--dep", action="store_true")
        parser.add_argument("--disc", action="store_true")
        parser.add_argument("--all", action="store_true")
        args = parser.parse_args()

        mode = "dep" if args.dep else "ind"
        if args.all:
            eeg_dir = os.path.join(root_dir, "svm_features_all")
        discrete_waveform = args.disc

        parameters = {"min_samples_split": [2,4,8,16], "min_samples_leaf": [2,4,8,16]}
        
        n_feats_list = [5,10,15,20,25,30,40,50,100]
        if args.test:
                n_feats_list = n_feats_list[:2]
                for param in parameters.keys():
                        parameters[param] = parameters[param][:1]       

        feats_paths = [os.path.join(eeg_dir,f) for f in os.listdir(eeg_dir)]
        audio_paths = [os.path.join(env_dir,f) for f in os.listdir(env_dir)]
        
        for path in feats_paths:
            if "05" in path or "26" in path:
                feats_paths.remove(path)
        for path in audio_paths:
            if "05" in path or "26" in path:
                audio_paths.remove(path)        

        grid_search = RegressorGridSearch(RandomForestRegressor, n_feats_list, mode=mode, discrete_waveform=discrete_waveform)
        grid_search.set_match_function(match_function)  
        grid_search.set_grid_params(parameters)
        grid_search.set_feats_paths(feats_paths)
        grid_search.set_audio_paths(audio_paths)
        grid_search.set_scorer(make_scorer(mean_squared_error, greater_is_better="False"))
        grid_search.set_data()
        grid_search.make_splits()
        grid_search.search_n_feats()    
