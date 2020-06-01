import numpy as np
import pandas as pd
import os
import subprocess
from utils import root_dir

results_dir = os.path.join(root_dir, "results")

if os.path.exists(os.path.join(results_dir, "results_df.pd")):
	df = pd.read_pickle(os.path.join(results_dir, "results_df.pd"))
	#df.to_pickle(os.path.join(root_dir,"scratchhome_j","results_backup.pd"))

results_list = []

for f in os.listdir(results_dir):
	if f.endswith("score.npy"):
		score_dict = np.load(os.path.join(results_dir,f),allow_pickle=True)[0]
		score_dict["filename"] = f
		results_list.append(score_dict)

all_vars = []

for result in results_list:
	for var in result.keys():
		all_vars.append(var)


all_vars = list(set(all_vars))

results_dict = {}


for k in all_vars:
	results_dict[k] = []

for result in results_list:
	for k in results_dict.keys():
		try:
			results_dict[k].append(result[k])
		except KeyError:
			results_dict[k].append("NA")

results_df = pd.DataFrame(results_dict)
print(results_df)

try:
	results_df["val"] = results_df["val"].astype(float)
except ValueError:
	pass

#print(results_df[["val","model","epochs","lr","dropout","nfeats","nrmse","filename"]].sort_values("nrmse")[:30])


results_df.to_pickle(os.path.join(results_dir, "results_df.pd"))

subprocess.call(["chmod","777",os.path.join(results_dir,"results_df.pd")])

#np.save(os.path.join(results_dir,"results_df"),results_df)
