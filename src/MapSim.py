"""
	Code to support in-silico simulation of nanomapping



	@Author: James Bannon
"""

""" Library Loads"""

import sys
import pickle
import numpy as np
import os
import re
from constants import *


"""
 			FUNCTION DEFINITIONS
"""
def print_patient(pid, nt, paren, bp):
	print("Patient Id Number:\t" +str(pid))
	print("Number of Translocations Simulated:\t"+str(nt))
	print("Parenthetic Structure of Simulated Translocations:\n")
	print("\t"+str(paren))
	print("Breakpoint sites and Insertion Lengths\n")
	print("\t"+str(bp))
"""
get_wc
	wrapper function for inserting a specific number of wildcard
		caracters into a string
"""
def get_wc(n):
	x = "."
	for i in range(n-1):
		x=x+"."
	return x

""""
fundoc: 
	
	From a specific alu, procedurally generate regex matches
	such that we can perform matching. 

	IN: 
		alu:= an alu or sequence that we're targeting with CRISPR/Cas-9
			  a <string>
		max_wild:= the maximum number of wild card characters
		scale := the ammount to


	OUT: 
		Patterns:= a dictionary where keys
				are regular expression objects (finite state automata)
				to be used for matching in map and values that are probabilities


"""
def load_templates(alu,max_wild=1,scale=1):
	templates = {}
	templates[re.compile(alu)]=0.0 # true match should have zero false-positive probability
	if max_wild+1>len(alu):
		""" abort condition""" 
		print("max number of wildcard characters longer than alu length")
		print("exiting")
		sys.exit(0) 
	for t in range(1,max_wild+1):
		for j in range(0,len(alu)-t+1):
			template = alu[0:j]+get_wc(t)+alu[j+t:]
			patt =re.compile(template)
			templates[patt]=scale*(1-np.exp(-t))
	return templates



"""
Match(X,Y,templates)

	Use a dictionary of regex FSAs to match strings. If a match is found we return
		true and the probability that it's a true match

"""

def match(Y,templates={}):
	for fsa in templates.keys():
		if fsa.match(Y):
			return True,templates[fsa];
	return False,0


"""
Creates a simulated 'consensus map' for a particular patient

	given an alu a sequence, a dictionary of templates, and a molecular length (measured in base pairs)
	 	 we return a consensus map of fragments in R^4
	 	 <c_i, p_i, l_i, sd_i>

"""
def make_consensus_map(alu, seq,templates,molecLen = 10**5):
	# performs nanomapping for a single patient
	
	start_loc = 0 #intialize window at left end
	cut_sites = [] #stores c_i
	lengths = [] # stores l_i
	cutprobs =[] # stores p_i
	stdevs = [] # stores sigma_i

	while start_loc< len(seq):
		if start_loc + molecLen>=len(seq):
			current_molecule = seq[start_loc:] # window molecule
			start_loc=len(seq) ## gives exit for while loop 
		else:
			current_molecule=seq[start_loc:start_loc+molecLen]
			start_loc + molecLen>=len(seq) #REMOVE
		
		## process current molecule, size indicated to meet machine limits
		curr=0 ##? set to startloc?
		while curr<len(current_molecule) and start_loc+curr<len(seq):
			dex = min(curr+len(alu),len(current_molecule))
			binding,p_i = match(current_molecule[curr:dex],templates)
			if binding:
				cut_sites.append(start_loc+curr)
				cutprobs.append(p_i)
				stdevs.append(np.random.uniform(MIN_SIGMA,MAX_SIGMA))
				curr +=len(alu)
			else:
				curr +=1;
		start_loc+=molecLen #increment to end while loop, redundant if start_loc + molecLen>=len(seq)
	cut_sites.append(len(seq))

	for j in range(len(cut_sites)-1):
		diff = cut_sites[j+1]-cut_sites[j]
		if(diff<0):
			cut_sites[j+1]-cut_sites[j]
			print("found a negative difference!")
			print(cut_sites[j+1])
			print(cut_sites[j])
			print(j)
			print(len(cut_sites))
		lengths.append(diff)
	
	map_ = []
	for c,p,l,s in zip(cut_sites,cutprobs,lengths,stdevs):
		tup = (c,p,l,s)
		map_.append(tup)
	return map_;
	
"""
	Produces a simulated insilico map for a given sequence and alu.
		the map consists of a vector of lengths a_j
		and represents the 'true' map.

		Used for both patient-level 'true' maps
		as well as mapping reference genomes
	"""
def make_insilico_map(alu, seq,templates,molecLen =10**5 ):
	
	cut_sites = [] #stores cut sites
	lengths = [] # stores lengths of fragments a_j

	start_loc = 0 #intialize window at left end
	# "bite off" a subsequence of the full sequence 
	# of a size that fits into the machine
	while start_loc< len(seq):
		if start_loc + molecLen>=len(seq):
			current_molecule = seq[start_loc:] # window molecule
			start_loc=len(seq) ## gives exit for while loop 
			#print('regime 2')
		else:
			current_molecule=seq[start_loc:start_loc+molecLen]
			#print('regime 1')
		## process current molecule, size indicated to meet machine limits
		curr=0 
		while curr<len(current_molecule) and start_loc+curr<len(seq):
			dex = min(curr+len(alu),len(current_molecule))
			binding,p_i = match(current_molecule[curr:dex],templates)
			if binding:
				cut_sites.append(start_loc+curr)
				"""print("curr = " +str(curr))
				print("molecule length = " + str(len(current_molecule)))
				print("adding " + str(start_loc+curr) + " to cut sites")
				print("index is " + str(dex))
				print("current molecule is \n" + str(current_molecule[curr:dex]))
				print("start_loc = "+str(start_loc))
				print("length of molecule = " + str(len(current_molecule)))
				print("length of sequence = " + str(len(seq)))"""
				curr +=len(alu)
			else:
				curr +=1;
		start_loc+=molecLen #increment to end while loop, redundant if start_loc + molecLen>=len(seq)
	cut_sites.append(len(seq))
	for j in range(len(cut_sites)-1):
		diff = cut_sites[j+1]-cut_sites[j]
		if(diff<0):
			print("found a negative difference!") # report error, don't abort
			print("index of error = "+str(j))
			print("number of cut sites = " + str(len(cut_sites)))
			print(diff)
			print(cut_sites[j])
			print(cut_sites[j+1])
			print(cut_sites[j:])
			print(len(seq))
			sys.exit(0)
		lengths.append(diff)
	return lengths; # since this is in silico true map, just keep lengths. 


		
"""
def readPatientFile(fname)
Reads in a sequence over the alphabet {A,T,C,G}		from a given file
"""

def readPatientFile(fname):
	"""TODO edit to reflect new status """
	#TODO: wrap in try-catch.
	patient = open(fname, 'r')
	count = 0;
	p_seq = ""
	for line in patient:
		if count==0:
			p_id = int(line.rstrip())
			count+=1
		elif count==1:
			num_tlocs = int(line.rstrip())
			count +=1
		elif count==2:
			parens = line
			count+=1
		elif count==3:
			blist = line
			count+=1
		else:
			p_seq = "".join( (p_seq,line))
	patient.close()
	p_seq = p_seq.replace("\n","")
	return p_id, num_tlocs, parens, blist, p_seq


"""
def cohort_map(alu,templates, src, dest)
	maps an entire cohort of patients
"""	



def main():
	print(MapSim_msg)
	
	""" Load in alu and create FSA templates for matching"""

	
	
	with open(ALU_SRC,"r") as f:
		alu = f.readline()
	templates_IS = load_templates(alu,1,1)
	templates_C= load_templates(alu,MAX_WILD,SCALE)
	

	print("Now creating in-silico reference maps")
	ref_seqs = [seq for seq in os.listdir(REF_SEQS) if not seq.startswith(".")]
	for fname in ref_seqs:
		print(fname)
		ref_file = open(REF_SEQS+fname,"r")
		label = ref_file.readline()
		label= label.replace("\n","")
		ref_seq=""
		for line in ref_file:
			ref_seq= "".join( (ref_seq,line))
		ref_seq = ref_seq.replace("\n","")
		ref_file.close()
		_map = make_insilico_map(alu,ref_seq,templates_IS)
		tup =(label,_map)
		pickle.dump(tup,open(REF_MAPS+label+"_ref_map.p","wb"))
	
	


	print("Now creating in-silico & consensus patient maps")
	
	patient_seqs = [seq for seq in os.listdir(PATIENT_SEQS) if not seq.startswith(".")]
	count=0;
	for fname in patient_seqs:
		patient_id, num_tlocs, parens, bplist, p_seq = readPatientFile(PATIENT_SEQS+fname)	
		print("patient "+str(count+1)+" out of "+str(len(patient_seqs)))
		print("in silico map")
		_map = make_insilico_map(alu,p_seq,templates_IS)
		print("consensus map")
		c_map = make_consensus_map(alu,p_seq,templates_C)
		i_tup =(patient_id,num_tlocs,parens, bplist, _map)
		c_tup =(patient_id, num_tlocs,parens, bplist, c_map)
		pickle.dump(i_tup,open(PATIENT_MAPS_IS+"patient_"+str(patient_id)+"_IS_map.p","wb"))
		pickle.dump(c_tup,open(PATIENT_MAPS_C+"patient_"+str(patient_id)+"_C_map.p","wb"))
		count+=1
		

if __name__ == '__main__':
	main()