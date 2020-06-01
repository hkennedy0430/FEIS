import numpy as np
from scipy.io import loadmat, wavfile
import os
import re
import pickle
import torch
#from mne.filter import filter_data
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from scipy.stats import iqr

root_dir = "/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/eeg_new"

def get_subject_data_dict(sub_dir,subject_number:int):
	"""Takes a sub-directory and subject number (e.g. "Natural_Speech", 1),
	 returns a dictionary of EEG recordings for a single subject

	(Keys are integers corresponding to runs)
	"""
	sd_path = os.path.join(root_dir,sub_dir)
	EEG_path = os.path.join(sd_path,"EEG")
	subj_name = "Subject" + str(subject_number)
	
	EEG_subj_path = os.path.join(EEG_path,subj_name)
	
	file_re_EEG = re.compile(r"Subject[0-9]+_Run([0-9]+).mat")

	EEG_rec_dict = {}
	
	for f in os.listdir(EEG_subj_path):
		file_match = re.match(file_re_EEG,f)
		if file_match:
			EEG_rec_dict[int(file_match[1])] = loadmat(os.path.join(EEG_subj_path,f))
		
	return(EEG_rec_dict)
	

def get_audio_data_dict(sub_dir):
	"""Takes in name of sub-directory, e.g. "Natural_Speech",
	returns a dictionary of audio recordings.

	(Keys are integers corresponding to runs)
	"""
	
	sd_path = os.path.join(root_dir,sub_dir)
	
	stimuli_path = os.path.join(sd_path,"Stimuli","Wav")

	file_re_audio = re.compile(r"audio([0-9]+).wav")
	
	audio_rec_dict = {}
	
	for f in os.listdir(stimuli_path):
		file_match = re.match(file_re_audio,f)
		if file_match:
			audio_rec_dict[int(file_match[1])] = wavfile.read(os.path.join(stimuli_path,f))
		
	return(audio_rec_dict)


def get_envelope_data_dict(sub_dir):
	"""Takes in name of sub-directory, e.g. "Natural_Speech",
	returns a dictionary of audio envelopes.

	(Keys are integers corresponding to runs)
	"""
		
	sd_path = os.path.join(root_dir,sub_dir)
	
	stimuli_path = os.path.join(sd_path,"Stimuli","Envelopes")

	file_re_audio = re.compile(r"audio([0-9]+)_128Hz.mat")
	
	audio_rec_dict = {}
	
	for f in os.listdir(stimuli_path):
		file_match = re.match(file_re_audio,f)
		if file_match:
			audio_rec_dict[int(file_match[1])] = loadmat(os.path.join(stimuli_path,f))
		
	return(audio_rec_dict)


################Nicked from StackOverflow###############################

from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    print(data.shape)
    y = filtfilt(b, a, data, axis=0)
    return y

###############end of pilfered section###############################

def bandpass_multi(data, lowcut, highcut,fs):
	out = []
	print(data.shape)
	for i in range(data.shape[1]):
		y = butter_bandpass_filter(data[:,i],lowcut,highcut,fs)
		out.append(y)
	out = np.vstack(out)
	return(out)


def subtract_mastoids(eeg_dict:dict):
	""" 'References' the EEG data by subtracting the mean value 
	of the two mastoid reference channels from all other 
	channels

	(Takes in and returns a dictionary)
	"""	
	new_dict = eeg_dict
	
	for p in eeg_dict.keys():
		eeg_data = eeg_dict[p]["eegData"]
		mastoids = eeg_dict[p]["mastoids"]
		mastoids = np.mean(mastoids,axis=1)
		mastoids = np.expand_dims(mastoids,1)
		mastoids = np.tile(mastoids,(1,128))
		new_data = eeg_data - mastoids
		new_dict[p]["eegData"] = new_data

	return(new_dict)



def get_filtered_eeg(eeg_data:np.ndarray):
	""" Takes in a single subject dictionary (where the keys are runs)
	Returns a dictionary of dictionaries, where the keys of the sub-
	dictionaries are EEG bands.
	"""

	filters_dict = {"delta":[0.5,4], "theta":[4,8], "alpha":[8,15],
			"beta":[15,32], "gamma":[32,63.9]}
	
	#eeg_data = eeg_data.T
	
	fs = 128

	filtered_data = {}
	
	for filt in filters_dict.keys():
		
		f = filters_dict[filt]		
		filtered = bandpass_multi(eeg_data,f[0],f[1],fs)
		

		filtered_data[filt] = filtered.T
	
	return(filtered_data)


def filter_eeg_dict(eeg_data:dict, single_bandpass=False):
	
	filtered_dict = {}

	fs = 128
	
	if single_bandpass==False:
		for run in eeg_data.keys():
			filtered_run = get_filtered_eeg(eeg_data[run]["eegData"])
			filtered_dict[run] = filtered_run
	else:
		for run in eeg_data.keys():
			filtered_dict[run] = bandpass_multi(eeg_data[run]["eegData"],0.5,63,128).T
			
	
	return(filtered_dict)


def get_eeg_mean_std(single_bandpass=False):
	"""gets the mean and std (per electrode) for all eeg data.
	Can be done by filters (single_bandpass=False) or for the
	singly bandpass-filtered data (sinlge_bandpass=True
	"""
	
	if single_bandpass == False:
		dict_name = "data_dict.pkl"
	else:
		dict_name = "single_bandpass.pkl"

	mean_dict = defaultdict(list)
	var_dict = defaultdict(list)

	mean_list = []
	var_list = []

	for s in range(1,20):	
		with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(s),dict_name), 'rb') as loadfile:
			eeg_dict = pickle.load(loadfile)
		
		for run in eeg_dict.keys():
			
			if single_bandpass == False:
				run_dict = eeg_dict[run]
				for filt in run_dict.keys():
					filt_mean = np.mean(run_dict[filt],axis=0)
					filt_var = np.var(run_dict[filt], axis=0)
					mean_dict[filt].append(filt_mean)
					var_dict[filt].append(filt_var)
			else:
				run_mean = np.mean(eeg_dict[run],axis=0)
				run_var = np.var(eeg_dict[run],axis=0)
				mean_list.append(run_mean)
				var_list.append(run_var)
	
	
	if single_bandpass == False:			
		for k in mean_dict.keys():
			mean_dict[k] = np.mean(mean_dict[k],axis=0)
			var_dict[k] = np.mean(var_dict[k],axis=0)
	
		std_dict = {}
		for k in var_dict.keys():
			std_dict[k] = np.sqrt(var_dict[k])

		return(mean_dict,std_dict)

	else: 
		means = np.mean(mean_list, axis=0)
		varz = np.mean(var_list, axis=0)
		
		stds = np.sqrt(varz)
		return(means,stds)		
		

def get_env_dict(norm=False, discrete=False):
	"""Convenience function for loading a (normalized or unnormalized) pickled dictionary of audio envelopes"""
	if norm == True:
		with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","envelope_dict_norm.pkl"), 'rb') as loadfile:
			env_dict = pickle.load(loadfile) 
	elif discrete == True:
		with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","discrete_dict.pkl"), 'rb') as loadfile:
			env_dict = pickle.load(loadfile) 
	else:
		with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","envelope_dict.pkl"), 'rb') as loadfile:
			env_dict = pickle.load(loadfile) 

	return env_dict

def save_env_dict(discretized=False):
	if discretized == False:
		raise NotImplementedError

#	else:
#		with 


def get_env_mean_std():
	"""Gets the mean and standard deviation of the audio envelope"""
	mean_list = []
	var_list = []

	env_dict = get_env_dict()
	
	for run in env_dict.keys():
		mean_list.append(np.mean(env_dict[run]))
		var_list.append(np.var(env_dict[run]))

	mean = np.mean(mean_list)
	std = np.sqrt(np.mean(var_list))
	
	return(mean,std)
				
def normalize_all_env():
	"""Normalizes all audio envelope files"""	
	env_dict = get_env_dict()

	env_mean, env_std = get_env_mean_std()

	norm_dict = {}
	
	for run in env_dict.keys():
		
		data = env_dict[run]

		normed_data = (data - env_mean)/env_std

		norm_dict[run] = normed_data

	with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","envelope_dict_norm.pkl"), 'wb') as savefile:
		pickle.dump(norm_dict, savefile)
	

def load_pickled_dict(subject_number:int, norm=False, single_bandpass=False):
	"""Convenience function for loading a pickled EEG dictionary"""
	if single_bandpass == True:
		with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(subject_number),"single_bandpass.pkl"), 'rb') as loadfile:
			eeg_dict = pickle.load(loadfile)
	elif norm == True:
		with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(subject_number),"normalized_dict.pkl"), 'rb') as loadfile:
			eeg_dict = pickle.load(loadfile)

	else:
		with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(subject_number),"data_dict.pkl"), 'rb') as loadfile:
			eeg_dict = pickle.load(loadfile)
	return(eeg_dict)

		
def pickle_dict(subject_number:int,dict_to_pickle:dict,save_name:str):
	"""Convenience function for pickling an EEG dictionary"""
	with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(subject_number),save_name), 'wb') as savefile:
		pickle.dump(dict_to_pickle, savefile)
	

def window_data(run_data,subj_id:int,run_id:int,w_len:int,stride:int,single_bandpass=False):
	subj_id = str.zfill(str(subj_id),2)
	run_id = str.zfill(str(run_id),2)
	run_id = str(run_id)
	
	if single_bandpass:
		all_filts_array = run_data  #type of run_data should be np.ndarray, not dict
		if type(all_filts_array) != np.ndarray:
			raise TypeError("This should be an array")
	else:
		all_filts_array = np.hstack([run_data[filt] for filt in run_data.keys()])

	
	array_len = len(all_filts_array)
	len_to_crop = array_len - (array_len % stride)
	cropped_array = all_filts_array[:len_to_crop].astype('float32')
	
	
	start_idx = 0
	w_id = 0	

	while start_idx + w_len < len(cropped_array):
		savename = os.path.join(win_dir,"EEG","w{0}_s{1}".format(w_len,stride),
			"s"+ subj_id+"r"+run_id+"w"+str.zfill(str(w_id),4))

		np.save(savename, cropped_array[start_idx:start_idx+w_len])
		start_idx += stride
		w_id += 1


def make_window_files(w_len:int,stride:int,single_bandpass=False):

	if not os.path.exists(os.path.join(win_dir, "EEG")):
		os.mkdir(os.path.join(win_dir, "EEG"))

	window_dir = os.path.join(win_dir,"EEG","w{0}_s{1}".format(w_len,stride))
	
	if not os.path.exists(window_dir):
		os.mkdir(window_dir)


	for subj in range(1,20):
		eeg_dict = load_pickled_dict(subj,norm=True, single_bandpass=single_bandpass)
		for run in range(1,21): 
			print(len(eeg_dict[run]))
			window_data(eeg_dict[run],subj,run,w_len,stride,single_bandpass=single_bandpass)

		
	
def make_window_files_env(w_len:int,stride:int):
	
	if not os.path.exists(os.path.join(win_dir, "Envelopes")):
		os.mkdir(os.path.join(win_dir, "Envelopes"))

	env_dict = get_env_dict(norm=True)
	
	window_dir = os.path.join(win_dir,"Envelopes","w{0}_s{1}".format(w_len,stride))
	
	if not os.path.exists(window_dir):
		os.mkdir(window_dir)

	for run in range(1,21):
		print(len(env_dict[run]))
		return
		run_id = str.zfill(str(run),2)
		run_array = env_dict[run]

		
		array_len = len(run_array)
		len_to_crop = array_len - (array_len % stride)
		cropped_array = run_array[:len_to_crop].astype("float32")
		
		window_list = []
		
		start_idx = 0
		w_id = 0	

		while start_idx + w_len < len(cropped_array):
			savename = os.path.join(window_dir,
					"r"+run_id+"w"+str.zfill(str(w_id),4))
			np.save(savename, cropped_array[start_idx:start_idx+w_len])
			start_idx += stride
			w_id += 1
	

def normalize_all(single_bandpass=False):
	mean_dict, std_dict = get_eeg_mean_std(single_bandpass=single_bandpass)
		
	if single_bandpass == False:
		out_name = "normalized_dict.pkl"
	else:
		out_name = "single_bandpass.pkl"
	

	for s in range(1,20):
		eeg_dict = load_pickled_dict(s)
		
		normalized_dict = {}	
				
		for run in eeg_dict.keys():
			
			if single_bandpass == False:
				
				run_dict = eeg_dict[run]
				normalized_dict[run] = {}
				
				for filt in run_dict.keys():
				
	
					unnormalized = run_dict[filt]
				
				
					tiled_means = np.tile(mean_dict[filt], (unnormalized.shape[0],1))
					tiled_std = np.tile(std_dict[filt], (unnormalized.shape[0],1))

					normalized_data = (unnormalized - tiled_means)/tiled_std
						
					normalized_dict[run][filt] = normalized_data
			else:
				unnormalized = eeg_dict[run] #not actually a dict
				tiled_means = np.tile(mean_dict, (unnormalized.shape[0],1))
				tiled_std = np.tile(std_dict, (unnormalized.shape[0],1))
				
				normalized_data = (unnormalized - tiled_means)/tiled_std
				normalized_dict[run] = normalized_data
		
		pickle_dict(s,normalized_dict,out_name)
		

def save_run_dicts():
	
	for s in range(1,20):
		eeg_dict = load_pickled_dict(s)
		
		for k in eeg_dict.keys():
			
			run_dict = eeg_dict[k]
			
			with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(s),"run{0}.split".format(k)), 'wb') as savefile:
				pickle.dump(run_dict, savefile)



def organize_data(single_bandpass=False):
	
	if single_bandpass == False:
		out_name = "data_dict.pkl"
	else:
		out_name = "single_bandpass.pkl"
		

	envelope_dict = get_envelope_data_dict("Natural_Speech")
	
	env_dict = {}
	
	for k in envelope_dict.keys():
		env_dict[k] = envelope_dict[k]["env"]

	#Clip audio envelopes where required:
		
	for s in range(1,20):
		eeg_dict = get_subject_data_dict("Natural_Speech",s)
		
		for k in sorted(eeg_dict.keys()):
			env_length = len(env_dict[k])
			eeg_length = eeg_dict[k]["eegData"].shape[0]
			if eeg_length < env_length:
				env_dict[k] = env_dict[k][:eeg_length]		

	
	with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","envelope_dict.pkl"), 'wb') as savefile:
		pickle.dump(env_dict, savefile) 

	#Clip eeg recordings to be the same length as audio envelopes

	for s in range(1,20):
		eeg_dict = get_subject_data_dict("Natural_Speech",s)
		eeg_dict = subtract_mastoids(eeg_dict) # Reference the data, subtracting the mastoid (reference) channels
							# (with the aim of removing electromagnetic interference)	
	
		for k in eeg_dict.keys():
			env_length = len(env_dict[k])
			eeg_length = eeg_dict[k]["eegData"].shape[0]
			eeg_dict[k]["eegData"] = eeg_dict[k]["eegData"][:env_length,:]		
			filtered_data = filter_eeg_dict(eeg_dict,single_bandpass=single_bandpass) #Carry out bandpass filtering
			

		with open(os.path.join(root_dir,"Natural_Speech","EEG","Subject{0}".format(s), out_name), 'wb') as savefile:
			pickle.dump(filtered_data, savefile)		


def basic_preprocessing(single_bandpass=False):
	""" If 'single bandpass' == True, we don't use the 5 bandpass filters.
		Otherwise, we do. 
			"""

	if single_bandpass == False:
		win_dir = "/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/windows" 
	else:
		win_dir = "/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/wins_1_filt" 
	
	organize_data(single_bandpass=single_bandpass) #This loads data from matlab files, clips all files to the correct length,
			#references the data (subtracts mastoid channels) and saves them as dictionaries. 
	normalize_all(single_bandpass=single_bandpass)
	make_window_files(256,246, single_bandpass=single_bandpass) #Segments the EEG data into 2-second chunks
	make_window_files_env(256,246)					#Does the same for the envelopes


def unnormalize_envelope(data):
	
	if type(data) == torch.Tensor:
		data = data.numpy()
	
	mean, std = get_env_mean_std()

	tiled_mean = np.tile(np.array([mean]), data.shape)

	tiled_std = np.tile(np.array([std]), data.shape)

	unnormed = (data * tiled_std) + tiled_mean

	return unnormed


def get_nrmse(model_output, target):
	"""Calcuates the Normalized Root Mean Squared Error on the model output by first unnormalizing the data,
	calculating MSE, then taking the square root and normalizing by the IQR (Interquartile Range) of 
	the targets"""
	
	if type(model_output) == torch.Tensor:
		model_output = model_output.numpy()
	
	if type(target) == torch.Tensor:
		target = target.numpy()

	model_output = model_output.reshape(-1)
	target = target.reshape(-1)

	unnormed_output = unnormalize_envelope(model_output)
	unnormed_target = unnormalize_envelope(target)
	
	mse = mean_squared_error(unnormed_target, unnormed_output)

	rmse = np.sqrt(mse)
	
	iq_r = iqr(unnormed_target)

	return rmse/iq_r	

def get_nrmse_torch(val_loader, net):
	net.eval()

	mse_list = []
	iqr_list = []
	
	for X, batch in val_loader:
		target = batch.numpy()
		model_output = net(X).detach().numpy()

		model_output = model_output.reshape(-1)
		target = target.reshape(-1)

		unnormed_output = unnormalize_envelope(model_output)
		unnormed_target = unnormalize_envelope(target)
	
		iqr_list.append(iqr(unnormed_target))
		mse_list.append(mean_squared_error(unnormed_target, unnormed_output))
	
	mse = np.mean(mse_list)
	iq_r = np.mean(iqr_list)

	rmse = np.sqrt(mse)
	
	return rmse/iq_r		


def discretize_waveform(only_return_values=False):
	
	first_quartile_list = []
	median_list = []
	third_quartile_list = []

	env_dict = get_env_dict(norm=False)
		
	for run in env_dict:
		first_quartile_list.append(np.percentile(env_dict[run],25))
		median_list.append(np.median(env_dict[run]))
		third_quartile_list.append(np.percentile(env_dict[run],75))		

	first_quartile = np.mean(first_quartile_list)
	second_quartile = np.mean(median_list)
	third_quartile = np.mean(third_quartile_list)


	first_quarter_means = []
	second_quarter_means = []
	third_quarter_means = []
	fourth_quarter_means = []


	discrete_dict = {}
	
	for run in env_dict.keys():
		r = env_dict[run]
			
		discrete_dict[run] = np.ones(r.shape)
		discrete_dict[run][np.where(r>first_quartile)] = 2
		discrete_dict[run][np.where(r>second_quartile)] = 3
		discrete_dict[run][np.where(r>third_quartile)] = 4

		first_quarter_means.append(np.mean(r[np.where(r<first_quartile)]))
		second_quarter_means.append(np.mean(r[np.where((r>first_quartile) & (r <second_quartile))]))
		third_quarter_means.append(np.mean(r[np.where((r>second_quartile) & (r <third_quartile))]))
		fourth_quarter_means.append(np.mean(r[np.where(r>third_quartile)]))
	
	if only_return_values == False:
		with open(os.path.join(root_dir,"Natural_Speech","Stimuli","Envelopes","discrete_dict.pkl"), "wb") as save_file:
			pickle.dump(discrete_dict, save_file)

	fiq_mean = np.mean(first_quarter_means)
	sq_mean = np.mean(second_quarter_means)
	tq_mean = np.mean(third_quarter_means)
	foq_mean = np.mean(fourth_quarter_means)


	ds = discrete_dict[1]
	proper_values = np.zeros(ds.shape)	
	proper_values[np.where(ds == 1)] = fiq_mean
	proper_values[np.where(ds == 2)] = sq_mean
	proper_values[np.where(ds == 3)] = tq_mean
	proper_values[np.where(ds == 4)] = foq_mean

	print(proper_values)

	if only_return_values == False:
		np.save("continuous_waveform", env_dict[1])
		np.save("discrete_waveform", proper_values)
	else:
		return([fiq_mean,sq_mean,tq_mean,foq_mean])


def get_proper_values(discretized_waveform:np.ndarray):
	proper_values = discretize_waveform(only_return_values=True)
	
	output_waveform = np.zeros(discretized_waveform.shape)
	
	output_waveform[np.where(discretized_waveform == 1)] = proper_values[0]	
	output_waveform[np.where(discretized_waveform == 2)] = proper_values[1]	
	output_waveform[np.where(discretized_waveform == 3)] = proper_values[2]	
	output_waveform[np.where(discretized_waveform == 4)] = proper_values[3]	

	return(output_waveform)

def channel_names_to_indices(channel_names:list):
	
	letters = ["A","B","C","D"]

	electrodes = [l + str(i) for l in letters for i in range(1,33)]
	
	return([electrodes.index(c.upper()) for c in channel_names])
	
	

if __name__ == "__main__":
		
	print(get_eeg_mean_std())
	
