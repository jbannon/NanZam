###
#
#	Ful NanZam
#
###

import sys
import pickle
import nanzam_utils as utils
import pandas as pd
import numpy as np
import os
import time
import bisect
from EBBS import *
from constants import *
from collections import Counter
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import math
import plot_utils as pltools



def featurize_lindsey(x,J=7):
	""" takes in a numpy array of shape (n,)
		and returns
		featurized_matrix
				 a 2d array of shape (n,J+1) 
				 where
				 featurized_matrix[i,j] = x[i]^j+1
				 j=0, ...j-1
				 featurized_matrix[i,j] = 1
				 j= J
				 """
	powers = np.arange(0,J+1) #exponents to raise x entries to
	if isinstance(x,float):
		featurized_matrix = x**powers
		return featurized_matrix
	elif len(x.shape)==0: #we pass a float
		featurized_matrix = x**powers
		return featurized_matrix
	else:
		featurized_matrix = np.zeros((x.shape[0],J+1)) # set up zeros
		for i in range(x.shape[0]):
			featurized_matrix[i] =x[i]**powers
		return featurized_matrix



def fit_mixture(scores, J=7,binwidth=0.1):
	
	l = np.amin(scores)
	u = np.amax(scores)


	b = np.arange(l,u,binwidth)
	h = np.histogram(scores,b)
	
	counts = h[0]
	bins= h[1]
	
	midpoints = (bins[1:]+bins[:-1])/2
	midpoint_powers = featurize_lindsey(midpoints)

	model = sm.GLM(counts, midpoint_powers, family=sm.families.Poisson())
	model =model.fit()
	return model

def get_densities(candidates, theo_Null = True, theo_Prior=True,p=0.99):
	scores = [x[3] for x in candidates]
	mean = np.mean(scores)
	std = np.std(scores)
	scores = (scores -mean)/std

	if theo_Null:
		f_0 = lambda x: norm.pdf(x,0,1)
	else:
		f_0=lambda x: norm.pdf(x,0,1) 
		"""TODO: make this fit empirically"""

	if theo_Prior:
		pi_0 = p
	else:
		pi_0 = p 
		"""TODO: make this fit empirically """
	f_mix  = fit_mixture(scores)
	return pi_0, f_0, f_mix,mean,std



def get_fdr(x,pi_0,f_0, f_mixture):
	"""
	x the point to get fdr
	f_0 the null distribution
	pi_0 the null hypothesis probability
	f the mixture distribution

	"""
	
	return (pi_0*f_0(x))/(f_mixture.predict(featurize_lindsey(x)))


def fdr_sort(candidates,w,thresh=0.2):
	"""TODO: empirical null """
	""" first fits fdr model"""
	""" then sorts array of scores based on fdr function"""
	pi_0, f_0, f_mixture, mu, sigma= get_densities(candidates)
	candidates = sorted(candidates, key=lambda cand: get_fdr((cand[3]-mu)/sigma,pi_0,f_0,f_mixture)[0])
	fdrs= [get_fdr((cand[3]-mu)/sigma,pi_0,f_0,f_mixture)[0] for cand in candidates]
	plt.plot(fdrs)
	plt.show()
	print(candidates)
	print(fdrs)
	sys.exit(0)
	meets_thresh_dex =[]
	for cand in candidates:
		meets_thresh_dex.append(get_fdr(cand[3],pi_0,f_0,f_mixture)[0])
	dex = np.where(np.array(meets_thresh_dex)<thresh, 1,0)
	print(dex)
	return candidates, dex
	#ordered = sorted(all_candidates, key=lambda tup:-tup[-1])



def printTable(HT):
	for k in HT.keys():
		print(k)
		print(HT[k])

def print_parens(paren_list):
	for struct in paren_list:
		if struct[0]=="(":
			print("(_"+struct[1]+" ",end="")
		elif struct[0]==")":
			print(struct[1]+"_) ",end="")
		else:
			print("(_"+struct[1]+" "+struct[1]+"_) ",end="")
	print("\n")

def top_candidate_seq(candidate_map):
	wt = len(Counter([x[0] for x in candidate_map[0]]).most_common())==1
	return wt

def majority_vote(candidate_list):
	wt_dex = list()
	wt_count = 0
	nwt_dex = list()
	nwt_count = 0
	for i in range(len(candidate_list)):
		candidate_map =candidate_list[i]
		wt = len(Counter([x[0] for x in candidate_map[0]]).most_common())==1
		if wt:
			wt_count+=1;
			wt_dex.append(i)
		else:
			nwt_count+=1
			nwt_dex.append(i)
	if wt_count>=nwt_count:
		return True, candidate_list[np.random.choice(wt_dex)]
	else:
		return False, candidate_list[np.random.choice(nwt_dex)]
def coinflip(p=0.98):
	val=np.random.uniform(0,1)
	if val<=p:
		return True
	else:
		return False

def fdr_candidate_process(candidate_list,thresh=0.05):
	scores = np.array([x[3] for x in candidate_list])
	#print(scores)
	zscores = (scores-np.mean(scores))/np.std(scores)
	#print(zscores)
	lengths = [len(x[0]) for x in candidate_list]
	#print(candidate_list[0][0])
	#if np.abs(np.mean(np.array(lengths))-29)<0.01:
		#return True, candidate_list[0]
	
	if math.isnan(zscores[0]):
		if len(np.unique(scores))==1:
			return coinflip(), candidate_list[0]
		else:
			return coinflip(), candidate_list[0]
	wt=True
	dex =-1;
	up =  norm.cdf(zscores)
	down = 1-norm.cdf(zscores)
	for i in range(len(zscores)):
		if up[i]<thresh or down[i]<thresh:
			wt=False
			dex=i;
	if not wt:
		return wt, candidate_list[0]
	else:
		return wt, candidate_list[0]


def handle_candidate_sequences(candidate_list,mode="fdr"):
	#print(candidate_list)
	#print(candidate_list[0][3])
	lengths = [len(x[0]) for x in candidate_list]
	#print(candidate_list[0][0])
	print("average length of candidate sequence is: "+str(np.mean(np.array(lengths))))

	if mode not in ["top","majority","fdr"]:
		print("invalid mode pasedd to handle_candidate_sequences")
		print("aborting")
		sys.exit(0)
	elif mode=="top":
		wt = top_candidate_seq(candidate_list[0])
		return wt, candidate_list[0]
	elif mode == "majority":
		wt,seq =majority_vote(candidate_list)
		return wt,seq
	elif mode =="fdr":
		wt,seq = fdr_candidate_process(candidate_list)
		return wt,seq


def patient_nanzam(mer_length,patient_tup,almap,HashTable,verbose=True,	mode = "fdr",beamwidth=5,fdr_counter=np.float('inf')):
	"""return correct count """
	

	if not callable(beamwidth):
		beamwidth = math.ceil(beamwidth)
		temp = beamwidth
		beamwidth= lambda x: temp

	if verbose:
		print("Patient id: " +str(patient_tup[0]))
		print("Number of Translocations: " + str(patient_tup[1]))
		print("Patient Is True Wild Type: " + str(patient_tup[1]==0))
		print("Patient Lengths : " + str(patient_tup[4]))
		p_id = patient_tup[0]
		is_wild_type = patient_tup[1]==0
		num_tlocs = patient_tup[1]
		patient_map = patient_tup[4]
	else:
		p_id = patient_tup[0]
		is_wild_type = patient_tup[1]==0
		num_tlocs = patient_tup[1]
		patient_map = patient_tup[4]
	translated_map = "".join(utils.translate2(patient_map,almap,jitter=True)[0])
	if verbose:
		s=time.time()
		candidate_sequences = EBBS(translated_map, HashTable,fdr_counter=fdr_counter,w=beamwidth)
		
		called_wild_type, final_seq = handle_candidate_sequences(candidate_sequences,mode=mode)
		e = time.time()
		print("Final Map: \n")
		print("\t",end="")
		print(final_seq[0])
		print("\n")
		print("parenthetic structure of final map:\n")
		print("\t",end="")
		print_parens(final_seq[2])
		print("took " + str(e-s) + " seconds.")
		timetook = e-s
		if (is_wild_type and called_wild_type):
			print("system called correctly, true negative")
		elif (not is_wild_type and not called_wild_type):
			print("system called correctly, true positive")
		print("\n")
		return is_wild_type,called_wild_type,timetook

	else:
		candidate_sequences = EBBS(translated_map, HashTable,w=lambda t: 5)
		called_wild_type, final_seq = handle_candidate_sequences(candidate_sequences,mode=mode)
		return is_wild_type,called_wild_type

def run_offline_phase(mer_length = 5,alu ="TGTAATCCCAGCACTTTGGGAGG",
	alphsize=8, table_size = 23,ref_path = REF_MAPS, add_two=True,D=256, epsilon = 10,
	normalize = False):
	

	HashMap = utils.makeTable(table_size)
	maps = [f for f in os.listdir(ref_path) if not f.startswith(".")]
	ref_maps = []
	lengths = []
	

	for map_file in maps:
		current_map = pickle.load(open(REF_MAPS+map_file,"rb"))
		ref_maps.append(current_map)
		lengths.extend(current_map[1])

	alphabet,almap, bins = utils.make_Alphabet(np.array(lengths),alphsize)
	
	pickle.dump(almap,open(SYSTEM_BINS+"almap.p","wb"))
	pickle.dump(alphabet, open(SYSTEM_BINS+"alph.p","wb"))
	pickle.dump(bins, open(SYSTEM_BINS+"bins.p","wb"))
	epsilons = [15,0.2]
	for ref_map,epsilon in zip(ref_maps,epsilons):
		label = ref_map[0]
		#print(ref_map[1])
		#print("hashing "+str(label))
		t_map = utils.translate2(ref_map[1],almap,jitter=False,epsilon=3)
		for i in range(len(t_map)-mer_length):
			mer="".join(t_map[i:i+mer_length])
			dex = utils.hash(mer,D,table_size)
			HashMap[dex].append( (label,i, mer))
	pickle.dump(HashMap, open(SYSTEM_BINS+"HashTable.p","wb"))
	printTable(HashMap)
	#sys.exit(0)
	

def run_online_phase(mer_length = 5,  
	patient_c = PATIENT_MAPS_C, 
	patient_is = PATIENT_MAPS_IS,
	in_silico=True,
	mode="fdr",
	verbose=True,D=256,w=5,fdr_counter=np.float('inf')):
	

	HashTable = pickle.load(open(SYSTEM_BINS+"HashTable.p","rb"))	
	alphabet = pickle.load(open(SYSTEM_BINS+"alph.p","rb"))
	bin_array = pickle.load(open(SYSTEM_BINS+"bins.p","rb"))
	almap = pickle.load(open(SYSTEM_BINS+"almap.p","rb"))
	

	if in_silico:
		patient_dir = patient_is
	else:
		patient_dir = patient_c

	patient_list = []
	for filename in [x for x in os.listdir(patient_dir) if not x.startswith(".")]:
		temp = pickle.load(open(patient_dir+filename,"rb"))
		patient_list.append(temp)
	neg_example_count = 0
	pos_example_count = 0
	true_type = list() # 1 = transloc, 0 = wild type
	called_type = list()
	k =0
	dem_thresh = 10;
	demo=False
	num_correct = 0
	times=[]
	for patient in patient_list:
		if verbose:
			if demo:
				if k>11:
					break;
				print("Patient " + str(k+1) +" out of "+str(dem_thresh+1))
			else:
				print("Patient " + str(k+1) +" out of "+str(len(patient_list)))
		k+=1
	
		true_wild_type, called_wild_type,duration = patient_nanzam(mer_length,patient, almap,HashTable,verbose=verbose,beamwidth=w,fdr_counter=fdr_counter)
		times.append(duration)
		if true_wild_type==called_wild_type:
			num_correct+=1
		
		if true_wild_type:
			neg_example_count+=1
			true_type.append(0)
			if called_wild_type:
				called_type.append(0)
			else:
				called_type.append(1)
		else:
			pos_example_count+=1
			true_type.append(1)
			if called_wild_type:
				called_type.append(0)
			else:
				called_type.append(1)
	cm = confusion_matrix(true_type, called_type)
	tn, fp, fn, tp = cm.ravel()	
	print()
	if demo and k>dem_thresh:
		print("percentage correct: "+str(((num_correct/(dem_thresh+1)))*100))
		#sys.exit(0)
	else:
		print("percentage correct: "+str(((num_correct/len(patient_list)))*100))
	print("making plot")
	pltools.plot_confusion_matrix(cm,classes = ["wild type", "translocated"],
		fig_name=FIG_PATH+"CONMAT_NANZAM.pdf")
	pltools.plot_confusion_matrix(cm,classes = ["wild type", "translocated"],
		fig_name=FIG_PATH+"CONMAT_NANZAM_norm.pdf",normalize=True)
	print("Average time took: "+ str(np.mean(times)))
	acc = (num_correct/len(patient_list))*100
	return tn,fp,fn,tp,cm,np.mean(times),acc
	
def run_full_nanzam():
	pass
	
def main():

	print(NANZAM_msg)
	print("offline phase")
	run_offline_phase()
	print("online phase")
	run_online_phase()
		


if __name__ == '__main__':
	main()
