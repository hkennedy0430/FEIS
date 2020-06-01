import numpy as np
import os
from scipy.signal import resample, butter, lfilter, filtfilt
from scipy.io import wavfile
from sklearn.kernel_approximation import Nystroem

root_dir = os.path.join(os.getcwd(),"feis") # anticipate that FEIS dir is child to (home) dir
filters_dict = {"delta":[0.5,4], "theta":[4,8], "alpha":[8,15], "beta":[15,32], "gamma":[32,63.9]}




def csv_to_numpy(csv_file_path:str,epoch_length:int):
	"""Takes a single .csv file and transforms it into a numpy array of
	shape (epochs, timesteps, channels).

	"epochs" here refers to experimental epochs, i.e. a single sound or 
	prompt. """

	data = np.genfromtxt(csv_file_path, delimiter=",")

	data = data[1:,2:16] #Removes header, and non-eeg csv columns 
	
	assert len(np.where(np.isnan(data))[0]) == 0, "There's a Nan in my data, what am I gonna do?"

	length_to_crop = len(data) % 256 #256 is the sampling frequency

	if length_to_crop != 0:
		data = data[:-length_to_crop]
	
	data = np.array(np.split(data, len(data)/(epoch_length*256)))

	return data



def make_numpy_files():
	
	epochs = [("stimuli",5),("thinking",5),("articulators",1),("speaking",5),("resting",5)]

	for speaker_n in range(1,22):
		
		speaker_n = str(speaker_n).zfill(2)
		
		for (epoch_type, epoch_length) in epochs:	
		
			file_path = os.path.join(root_dir,"experiments",speaker_n,"{}.csv".format(epoch_type))
	
			data = csv_to_numpy(file_path, epoch_length)
			
			save_path = os.path.join(root_dir,"experiments",speaker_n,"{}.npy".format(epoch_type))

			np.save(save_path,data)



def downsample_eeg():
	""" Overwrites the previous files to save space, so be careful with it"""
		
	
	epochs = [("stimuli",5),("thinking",5),("articulators",1),("speaking",5),("resting",5)]

	for speaker_n in range(1,22):
		
		speaker_n = str(speaker_n).zfill(2)
		
		for (epoch_type, epoch_length) in epochs:	
		
			file_path = os.path.join(root_dir,"experiments",speaker_n,"{}.npy".format(epoch_type))
	
			data = np.load(file_path)
			
			data = resample(data,640,axis=1) #This will mess up "articulators" but nvm

			save_path = os.path.join(root_dir,"experiments",speaker_n,"{}.npy".format(epoch_type))
		
			print(data.shape)
			
			np.save(save_path,data)



def downsample_wavs():
	
	for speaker_n in range(1,22):
		
		speaker_n = str(speaker_n).zfill(2)

		wav_dir = os.path.join(root_dir,"wavs",speaker_n,"combined_wavs")
				
		if os.path.exists(wav_dir):

			wav_paths = [os.path.join(wav_dir,x) for x in os.listdir(wav_dir) if x.endswith(".wav")]
		
			for wav in wav_paths:
			
				data = wavfile.read(wav)[1]
		
				data = resample(data, 640)

				np.save(wav[:-4],data)



def index_to_label(index:int):
	""" Takes in an index for a chunk of EEG data (an "epoch") and returns the associated 
	phonological label
	"""
	
	label_list = []

	with open(os.path.join(root_dir,"labels.txt")) as label_file:
		for line in label_file.readlines():
			label_list.append(line.strip("\n"))

	return(label_list[index])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq	#We divide by the nyquist frequency because butter() requires values between 0 and 1
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def put_windows_in_dir(speaker_n, epoch="stimuli", single_bandpass=False):

	speaker_n = str(speaker_n).zfill(2)

	if single_bandpass:
		eeg_windows = os.path.join(root_dir,"single_bandpass","windows","EEG")
		envelope_windows = os.path.join(root_dir,"single_bandpass","windows","Envelopes")
	else:
		eeg_windows = os.path.join(root_dir,"frequency_bands","windows","EEG")
		envelope_windows = os.path.join(root_dir,"frequency_bands","windows","Envelopes")

	hearing_file = os.path.join(root_dir,"experiments",speaker_n,"{0}.npy".format(epoch)) #File with EEG data of people hearing voice recordings
	wavs_dir = os.path.join(root_dir,"wavs",speaker_n,"combined_wavs") #File with ....aaa....aaa...aaa... style recordings

	if not os.path.exists(wavs_dir):
		print("No wavs")
		return None

	wavs_dict = {}

	wavs_list = [x for x in os.listdir(wavs_dir) if x.endswith(".npy")]

	for wav in wavs_list:
		data = np.load(os.path.join(wavs_dir,wav))
		wavs_dict[wav[:-4]] = data

	data = np.load(hearing_file)	
	
	win_no = 0

	if epoch == "thinking": 
		speaker_n = str(int(speaker_n) + 21) #Give a unique name to each file
	
	for i in range(len(data)):
		start_idx = 0

		label = index_to_label(i)    #Get the correct recording corresponding to the label
		audio_chunk = wavs_dict[label]	#of the EEG recording segment

		eeg_chunk = data[i] #A single epoch

		if single_bandpass == True:
			eeg_chunk = butter_bandpass_filter(eeg_chunk, 0.5,63,128)

		else:
			combined_chunk = []
			for key in filters_dict.keys():
				low, high = filters_dict[key]
				combined_chunk.append(butter_bandpass_filter(eeg_chunk, low, high, 128))
			eeg_chunk = np.hstack(combined_chunk)

		#Here is where we should do bandpass filtering etc

		for j in range(3):  # We split each chunk of data into three windows
			window_code = "s{0}w{1}".format(speaker_n,str(win_no).zfill(4))
			end_idx = start_idx + 256

			eeg_window = eeg_chunk[start_idx:end_idx]
			eeg_save_path = os.path.join(eeg_windows, window_code)
			np.save(eeg_save_path, eeg_window)

			audio_window = audio_chunk[start_idx:end_idx] #Actually the audio envelope, not raw audio
			
			if len(audio_window) < 256:
				import pdb; pdb.set_trace()
			envelope_save_path = os.path.join(envelope_windows, window_code)
			np.save(envelope_save_path, audio_window)

			win_no += 1
			start_idx += 192 #For chunks of data with len 640, we can use a window length of 200 and a 
					# stride of 150 to split the chunk into three windows
		


def combine_wav_files(speaker_n,epoch="stimuli"):
	
	speaker_n = str(speaker_n).zfill(2)	

	eeg_file = os.path.join(root_dir,"experiments",speaker_n,"{0}.npy".format(epoch)) #File with EEG data of people hearing voice recordings

	wavs_dir = os.path.join(root_dir,"wavs",speaker_n,"combined_wavs") #File with ....aaa....aaa...aaa... style recordings
	
	if not os.path.exists(wavs_dir):
		print("No wavs")
		return None
	
	wavs_dict = {}
	wavs_list = [x for x in os.listdir(wavs_dir) if x.endswith(".npy")]
	for wav in wavs_list:
		data = np.load(os.path.join(wavs_dir,wav))
		wavs_dict[wav[:-4]] = data

	data = np.load(eeg_file)

	win_no = 0

	if epoch == "thinking":
		speaker_n = str(int(speaker_n) + 21) #Give a unique name to each file

	all_chunks = []

	for i in range(len(data)):
		label = index_to_label(i)    #Get the correct recording corresponding to the label
		audio_chunk = wavs_dict[label]
		all_chunks.append(audio_chunk)

	envelope = np.concatenate(all_chunks)

	print(envelope.shape)
	np.save(os.path.join(root_dir,"combined_envelopes",speaker_n),envelope)


def normalize_envelopes():
	envs_dir = os.path.join(root_dir, "combined_envelopes")
	envelope_files = []
	for f in os.listdir(envs_dir):
		if f.endswith(".npy"):
			envelope_files.append(os.path.join(envs_dir, f))

	for e in envelope_files:
		env = np.load(e)
		env_mean = np.mean(env)
		env_std = np.std(env)
		norm_env = (env - env_mean)/env_std
		np.save(e, norm_env)

def normalize_svm_feats():
	feats_dir = os.path.join(root_dir, "svm_features")
	feats_files = []	

	for f in os.listdir(feats_dir):
		if f.endswith(".npy"):
			feats_files.append(os.path.join(feats_dir, f))
	
	for f in feats_files:
		feats = np.load(f)
		means_list = []
		stds_list = []
		for col in range(feats.shape[1]):
			means_list.append(np.mean(feats[:,col]))
			stds_list.append(np.std(feats[:,col]))
		
		array_len = len(feats)
		means_array = np.tile(means_list,(array_len,1))
		stds_array = np.tile(stds_list,(array_len,1))

		norm_feats = (feats - means_array)/stds_array
		np.save(f, norm_feats)


def do_kernel_approximation():
	feats_dir = os.path.join(root_dir, "svm_features")
	kernel_dir = os.path.join(root_dir, "kernel_features")
	feats_01 = np.load(os.path.join(feats_dir,"svm_feats_01.npy"))	
	feat_map = Nystroem()
	print("fitting")
	feat_map.fit(feats_01)	

	feats_files = []
	for f in os.listdir(feats_dir):
		if f.endswith(".npy"):
			feats_files.append(f)

	for f in feats_files:
		print("transforming file {0}".format(f))
		feats = np.load(os.path.join(feats_dir,f))
		transformed = feat_map.transform(feats)
		np.save(os.path.join(kernel_dir,f), transformed)
		


def do_kernel_approximation_all():
	feats_dir = os.path.join(root_dir, "svm_features_all")
	kernel_dir = os.path.join(root_dir, "kernel_features_all")
	feats_01 = np.load(os.path.join(feats_dir,sorted(os.listdir(feats_dir))[0]))	
	feat_map = Nystroem()
	print("fitting")
	feat_map.fit(feats_01)	

	feats_files = []
	for f in sorted(os.listdir(feats_dir)):
		if f.endswith(".npy"):
			feats_files.append(f)

	for f in feats_files:
		print("transforming file {0}".format(f))
		feats = np.load(os.path.join(feats_dir,f))
		transformed = feat_map.transform(feats)
		np.save(os.path.join(kernel_dir,f), transformed)


if __name__ == "__main__":
	do_kernel_approximation_all()
		


