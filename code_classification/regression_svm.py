import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import string
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
#from make_features import make_simple_feats
from utils import root_dir
from eeg_new_utils import get_env_dict, get_nrmse, unnormalize_envelope, get_proper_values
import datetime


env_dir = os.path.join(root_dir, "combined_envelopes")
eeg_dir = os.path.join(root_dir, "svm_features")
results_dir = os.path.join(root_dir,"results")	

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def window_data(data:np.ndarray):
	"""windows the data and reshapes it into 1D arrays"""	
	
	w_len = 128

	data_len = len(data)

	windowed_data = []

	#for i in range(len(data) - 128):
	#	windowed_data.append(data[i:i+128])
	
	for i in range(w_len):
		to_split = data[i:]
		l_to_crop = len(to_split) % w_len
		final_length = len(to_split) - l_to_crop
		cropped = data[:final_length]
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
		
	print(ordered_windows.shape)
	
	return ordered_windows
	
	

def get_feats(data:np.ndarray):
	windowed = window_data(data)
	return make_simple_feats(windowed)
		


def dict_to_array(data_dict:dict, data_type="EEG"):
	
	data_list = []

	data_list = get_feats(data_dict[1])

	for k in range(2,21):
 
		data = data_dict[k]

		if data_type == "EEG":
			windowed_feats = get_feats(data)		
			data_list = np.vstack((data_list,windowed_feats))
		else:
			data_list.append(data)
	
	print(data_list)
	return(np.vstack(data_list))


def get_correlations(X_array:np.ndarray, y_array:np.ndarray):

	correlation_list = []	
		
	for i in range(X_array.shape[1]):
		correlation_list.append(np.corrcoef(X_array[:,i],y_array.reshape(-1))[0,1])
	
	return correlation_list

def all_correlations():

	if args.broca:
		mid_envelope = 3
	else:
		mid_envelope = 21


	coefs_array = []
	env_dict = get_env_dict()
	for subject in [1]: #range(1,20)
				
		for run in range(1,21):
			eeg_feats = np.load(os.path.join(feats_dir, "Subject{0}".format(str(subject)),"svmfeats{0}.npy".format(str(run))))
			envelope = env_dict[run]
			cropped = envelope[mid_envelope:]
			cropped = cropped[:len(eeg_feats)]
			eeg_feats = eeg_feats[:len(cropped)]
			coefs_array.append(get_correlations(eeg_feats,cropped))

	coefs_array = np.array(coefs_array) # check this is working

	print(coefs_array.shape)

	coefs_array = np.arctanh(coefs_array) #get the z-score

	mean_coefs = np.mean(coefs_array, axis=0)

	abs_coefs = np.abs(mean_coefs)

	#import pdb; pdb.set_trace()

	order_indices = np.flip(np.argsort(abs_coefs))

	return(order_indices)

def get_feats(n_feats:int,order_indices:np.ndarray,test_mode=False):
	
	if args.broca:
		mid_envelope = 3
	else:
		mid_envelope = 21

	top_inds = order_indices[:n_feats]

	X_out = []
	y_out = []

	env_dict = get_env_dict(discrete=True)
	
	if test_mode == True:
		for run in range(1,21):
			X_run = np.load(os.path.join(feats_dir, "Subject1", "svmfeats{0}.npy".format(str(run)))) 
			X_run = X_run[:,top_inds]
			y_run = env_dict[run]
			y_run = y_run[mid_envelope:] # Each of our samples should correspond to one from the centre of the 
							#initial window (length 41)
			y_run = y_run[:len(X_run)]
			X_run = X_run[:len(y_run)]
			X_out.append(X_run)
			y_out.append(y_run)
	else:

		for subject in range(1,20):
		
			for run in range(1,21):
					
				X_run = np.load(os.path.join(feats_dir, "Subject{0}".format(str(subject)), "svmfeats{0}.npy".format(str(run)))) 
				X_run = X_run[:,top_inds]
				y_run = env_dict[run]
				y_run = y_run[mid_envelope:] # Each of our samples should correspond to one from the centre of the 
								#initial window (length 41)
				y_run = y_run[:len(X_run)]
				X_run = X_run[:len(y_run)]
				X_out.append(X_run)
				y_out.append(y_run)

	X_out = np.vstack(X_out)
	y_out = np.concatenate(y_out).squeeze().astype(np.int32)

	return(X_out, y_out)

def get_best_score(regressor):

	if regressor == None:
		return(10000)
	
	else:
		return regressor.best_score_


def match_function(x_file_path, y_file_path):
	only_ints = str.maketrans("","",string.ascii_lowercase + string.punctuation)
	if x_file_path.translate(only_ints) == y_file_path.translate(only_ints):
		return True
	else:
		return False


class RegressorGridSearch():
	def __init__(self, regressor_class, n_feats_list:list, test=False, discrete_waveform=False, smooth_waveform=True, mode="ind"):
		self.n_feats_list = n_feats_list
		self.discrete_waveform = discrete_waveform
		self.test = test
		self.smooth = smooth_waveform
		self.regressor_class = regressor_class
		self.regressor_name = self.regressor_class.__name__
		if self.test == True:
			self.regressor_name += "_test"
		if self.smooth:
			self.regressor_name += "_smooth"
		self.mode = mode
		self.best_score = 100000
		self.best_params = None
		self.best_n_feats = None
		self.checkpoint = None
		self.n_feats_done = []
		self.X = []
		self.y = []


	def set_scorer(self, scorer):
		self.scorer = scorer

	def set_grid_params(self, params:dict):
		self.grid_params = params

	def set_feats_paths(self, paths):
		self.feats_paths = sorted(paths)
								
	def set_audio_paths(self, paths):
		self.audio_paths = sorted(paths)

	def check_equal_length(self):
		if len(self.audio_paths) != len(self.feats_paths):
			print(self.audio_paths)
			print(self.feats_paths)
			raise ValueError("Number of audio clips should be the same as eeg clips")

	def set_regressor(self):
		self.regressor = GridSearchCV(self.regressor_class(), self.grid_params, cv=5, verbose=200, scoring=self.scorer, n_jobs=-1)

	def set_match_function(self, fn):
		self.match_function = fn

	def set_data(self):
		self.data_name = os.path.split(os.path.split(self.feats_paths[0])[0])[1] #gets name of lowest level directory
		if self.mode == "dep":
			self.feats_paths = self.feats_paths[:1]
			self.audio_paths = self.audio_paths[:1]		

		for i in range(len(self.feats_paths)):
			if not self.match_function(self.feats_paths[i], self.audio_paths[i]):
				import pdb; pdb.set_trace()
				raise ValueError("EEG and audio don't match")
			self.X.append(np.load(self.feats_paths[i]).astype(np.float32))
			self.y.append(np.load(self.audio_paths[i]).astype(np.float32))
		
		len_diff = len(self.y[0]) - len(self.X[0])   #Crop the y array (which is longer from the effect of 
		to_crop = int(len_diff/2)			#windowing the x data

		for i in range(len(self.y)):
			hello = self.y[i]
			self.y[i] = self.y[i][to_crop:-to_crop]
			if len(self.y[i]) != len(self.X[i]):
				import pdb; pdb.set_trace()
				raise ValueError("X and y arrays have different lengths")
				#import pdb; pdb.set_trace()

		if self.discrete_waveform:
			self.discretize_waveform()

		if self.smooth:
			self.smooth_waveform()	

	def discretize_waveform(self):
		y = np.concatenate(self.y)		
		first_quartile = np.percentile(y,25)
		median = np.median(y)
		third_quartile = np.percentile(y,75)
		
		for i in range(len(self.y)):
			self.y[i] = np.ones(len(self.y[i]))
			self.y[i][np.where(y[i]>first_quartile)] = 2
			self.y[i][np.where(y[i]>median)] = 3
			self.y[i][np.where(y[i]>third_quartile)] = 4

	def smooth_waveform(self):

		for i in range(len(self.y)):
			self.y[i] = smooth(self.y[i], 20000)


	def make_splits(self):
		
		if self.mode == "ind":
			self.X_train = self.X[:-2]
			self.y_train = self.y[:-2]

			self.X_val = self.X[-2]	
			self.y_val = self.y[-2]
	
			self.X_test = self.X[-1]
			self.y_test = self.y[-1]

			self.X_train = np.vstack(self.X_train)
			self.y_train = np.concatenate(self.y_train)

		elif self.mode == "dep":
			X = self.X[0]
			y = self.y[0]

			X_length = len(X)
			if X_length != len(y):
				raise ValueError
			
			self.X_train = X[:int(X_length*0.8)]
			self.y_train = y[:int(X_length*0.8)]
		
			self.X_val = X[int(X_length*0.8):-int(X_length*0.1)]			
			self.y_val = y[int(X_length*0.8):-int(X_length*0.1)]			

			self.X_test = X[-int(X_length*0.1):]
			self.y_test = y[-int(X_length*0.1):]
	
	def fit(self, n_feats):
		X_train = self.X_train[:,self.order_indices[:n_feats]]
		y_train = self.y_train
		self.regressor.fit(X_train, y_train)
	
	def update(self, n_feats):
		if self.regressor.best_score_ < self.best_score:
			self.best_score = self.regressor.best_score_
			self.best_params = self.regressor.best_params_
			self.best_regressor = self.regressor
			self.best_n_feats = n_feats
		


	def save(self):
		copy = RegressorGridSearch(self.regressor_class, self.n_feats_list)
		copy.best_regressor = self.best_regressor
		copy.test = self.test
		copy.regressor_class = self.regressor_class
		copy.regressor_name = self.regressor_name
		copy.mode = self.mode
		copy.best_score = self.best_score
		copy.best_params = self.best_params
		copy.best_n_feats = self.best_n_feats
		copy.checkpoint = self.checkpoint
		copy.n_feats_done = self.n_feats_done
		np.save(os.path.join(root_dir, "j", "regressors", self.regressor_name + "_" + self.data_name + "_" + self.mode), np.array([copy]))

	def get_most_correlated(self):
		coefs_list = []
		for i in range(len(self.X)):
			coefs_list.append(get_correlations(self.X[i],self.y[i]))
		coefs_array = np.array(coefs_list)
		coefs_array = np.arctanh(coefs_array) #get the z-score
		mean_coefs = np.tanh(np.mean(coefs_array, axis=0))
		abs_coefs = np.abs(mean_coefs)
		self.order_indices = np.flip(np.argsort(abs_coefs))


	def search_n_feats(self):	
		self.checkpoint_name = os.path.join("regressors", self.regressor_name + "_" + self.data_name + "_" + self.mode + ".npy")
		if os.path.exists(self.checkpoint_name):
			checkpoint = np.load(self.checkpoint_name, allow_pickle=True)[0]
			if len(checkpoint.n_feats_list) == 0:
				print("starting from scratch")
			else:
				self.checkpoint = checkpoint


		if self.checkpoint:
			self.n_feats_list = self.checkpoint.n_feats_list
			self.n_feats_done = self.checkpoint.n_feats_done
			self.best_score = self.checkpoint.best_score
			self.best_params = self.checkpoint.best_params		
			self.best_n_feats = self.checkpoint.best_n_feats
			self.best_regressor = self.checkpoint.best_regressor	
		
		self.get_most_correlated()

		while len(self.n_feats_list) > 0:
			n_feats = self.n_feats_list.pop(0)
			print("#" * 30)
			print("Starting grid search")
			print("params: {0}".format(self.grid_params))
			print("n_feats: {0}".format(n_feats))
			print("#" * 30)
			self.set_regressor()
			self.fit(n_feats)
			self.update(n_feats)
			self.n_feats_done.append(n_feats)
			self.save()
		
		self.save_results()

	def save_results(self):
		

		y_val_pred = self.best_regressor.predict(self.X_val[:,self.order_indices[:self.best_n_feats]])	
		now = datetime.datetime.now()
	
		out_dict = self.best_params
		nrmse = get_nrmse(y_val_pred, self.y_val)
	
		out_dict["model"] = self.regressor_class
		out_dict["val"] = abs(self.best_score)
		out_dict["nrmse"] = nrmse	
		out_dict["nfeats"] = self.best_n_feats	
		out_dict["input"] = self.data_name

		right_now = now.strftime("%m%d%H:%M")
	
		save_name = "{0}_{1}_{2}_".format(self.mode, self.regressor_name, right_now)
	
		score_name = os.path.join(results_dir, save_name + "score.npy")
		y_true_name = os.path.join(results_dir, save_name + "true.npy")	
		y_pred_name = os.path.join(results_dir, save_name + "pred.npy")

		np.save(score_name, np.array([out_dict]))
		np.save(y_true_name, self.y_val)
		np.save(y_pred_name, y_val_pred)
			

if __name__ == "__main__":
	
	n_feats_list = [5,10,15,20,25,30,50,100]
	parameters = {"C": [0.1,1,10], "gamma": [0.01,0.1,1]}
	feats_paths = [os.path.join(eeg_dir,f) for f in os.listdir(eeg_dir)]
	audio_paths = [os.path.join(env_dir,f) for f in os.listdir(env_dir)]
	
	for path in feats_paths:
		if "05" in path or "26" in path:
			feats_paths.remove(path)
	for path in audio_paths:
		if "05" in path or "26" in path:
			audio_paths.remove(path)	

	grid_search = RegressorGridSearch(SVR, n_feats_list)
	
	grid_search.data_name = os.path.split(os.path.split(feats_paths[0])[0])[1] #gets name of lowest level directory
	grid_search.save()
	grid_search.set_match_function(match_function)	
	grid_search.set_grid_params(parameters)
	grid_search.set_feats_paths(feats_paths)
	grid_search.set_audio_paths(audio_paths)
	grid_search.set_scorer(make_scorer(mean_squared_error, greater_is_better="False"))
	grid_search.set_data()
	grid_search.make_splits()
	grid_search.search_n_feats()	

