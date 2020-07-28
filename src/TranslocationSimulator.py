from constants import *
import os
import sys
import numpy as np
import math

""" 
	supported:
			if wild type and fixed return target, empty parens
			if translocations >1 return random tlocs and parens
	TODO:
		if wild type and random, return random, empty parens
		if wild type and not random, return random tloc and parens
"""



def splice_fixed(num_tlocs, src_seq,src_lab, targ_seq, targ_lab):
	bplist = []
	breakpoints = sorted(np.random.choice(np.arange(1,len(targ_seq)),num_tlocs,replace=False))
	spans = breakpoints
	spans.insert(0,0)
	spans.append(len(targ_seq))
	if len(breakpoints)==1:
		bp_loc = breakpoints[0]
		parens = "(_"+targ_lab+" " +"(_"+src_lab+" "+src_lab+"_)"+" "+targ_lab+"_)"
		insertion_length = math.ceil(np.random.random()*MAX_INSERT_LENGTH)
		cut_site = np.random.choice(np.arange(0,len(src_seq)))
		insertion_length = min(cut_site + insertion_length, len(src_seq))
		bplist.append((bp_loc,insertion_length))
		seq = targ_seq[:bp_loc] + src_seq[cut_site:cut_site+insertion_length] + targ_seq[bp_loc:]
		return seq, parens, bplist
	else:
		target_chunks = []
		seq = []
		parens = "(_"+targ_lab
		for dex in range(len(spans)-1):
			target_chunks.append(targ_seq[spans[dex]:spans[dex+1]])
		
		src_chunks = []
		for i in range(len(breakpoints)):
			bp_loc = breakpoints[i]
			insertion_length = math.ceil(np.random.random()*MAX_INSERT_LENGTH)
			cut_site = np.random.choice(np.arange(0,len(src_seq)))
			insertion_length = min(cut_site + insertion_length, len(src_seq))
			bplist.append((bp_loc,insertion_length))
			src_chunks.append(src_seq[cut_site:cut_site+insertion_length])
		
		for target_chunk, src_chunk in zip(target_chunks,src_chunks):
			seq.append(target_chunk)
			seq.append(src_chunk)
			parens += "(_"+src_lab+" " +src_lab+"_)"
		seq = "".join(seq)
		parens +=  targ_lab+"_)"
		return seq, parens,bplist 


def translocate_patient(num_tlocs, src, targ, mode,refs):

	mode = mode.lower()
	if mode not in ['random','fixed']:
		print("Invalid mode of translation; aborting")
		sys.exit(0)
	if num_tlocs == 0 and mode == 'fixed':
		return refs[targ], "()", "NA"
	elif num_tlocs>0 and mode=='fixed':
		seq, p, bl = splice_fixed(num_tlocs,refs[src],src,refs[targ],targ)
		return seq, p, bl
	else:
		print("invalid parameters; aborting")
		sys.exit(0)


def get_references(ref_dir=REF_SEQS):
	ref_dict = {}
	ref_seqs = [seq for seq in os.listdir(REF_SEQS) if not seq.startswith(".")]
	for fname in ref_seqs:
		ref_file = open(REF_SEQS+fname,"r")
		label = ref_file.readline()
		label= label.replace("\n","")
		ref_seq=""
		for line in ref_file:
			ref_seq= "".join( (ref_seq,line))
		ref_seq = ref_seq.replace("\n","")
		ref_file.close()
		ref_dict[label]=ref_seq
	return ref_dict

def write_wild_type(pid, n_tloc, paren,bp_list,seq):
	fname = PATIENT_SEQS + "patient_"+str(pid)+".txt"
	fstream = open(fname,"w")
	fstream.write(str(pid)+"\n")
	fstream.write(str(n_tloc)+"\n")
	fstream.write(paren+"\n")
	fstream.write(str(bp_list)+"\n")
	fstream.write(seq)
	fstream.close()


def write_variant_type(pid, n_tloc, paren,bp_list,seq):
	fname = PATIENT_SEQS + "patient_"+str(pid)+".txt"
	fstream = open(fname,"w")
	fstream.write(str(pid)+"\n")
	fstream.write(str(n_tloc)+"\n")
	fstream.write(paren+"\n")
	fstream.write(str(bp_list)+"\n")
	fstream.write(seq)
	fstream.close()

def print_patient(pid, nt, paren, bp):
	print("Patient Id Number:\t" +str(pid))
	print("Number of Translocations Simulated:\t"+str(nt))
	print("Parenthetic Structure of Simulated Translocations:\n")
	print("\t"+str(paren))
	print("Breakpoint sites and Insertion Lengths\n")
	print("\t"+str(bp))

def main():

	"""TODO: ARG SUPPORT"""

	print(tloc_sim_msg)
	
	num_patients = 1
	num_patients = 500 
	#expected_translocations = 10**-3 #2
	expected_translocations = 1
	a= 0
	b= 2*expected_translocations+1
	support = np.arange(a,b)
	
	ref_dict = get_references()

	params = {"src":'igh',"targ":'chr8',"mode":"fixed","refs":ref_dict,"num_tlocs":0}
	wt_count = 0
	v_count = 0
	
	q,e,r=0,0,0
	for i in range(num_patients):
		p_id = i;
		num_breakpoints = np.random.choice(support,p=np.array([90,6,4])/100)
		#print(num_breakpoints)
		if num_breakpoints==0:
			q=q+1;
		elif num_breakpoints==1:
			e+=1
		else:
			r+=1
		params["num_tlocs"]=num_breakpoints
		seq, paren_struc, bp_list = translocate_patient(**params)
		wild_type = num_breakpoints==0
		
		if wild_type:
			write_wild_type(p_id, num_breakpoints,paren_struc,bp_list,seq)
			wt_count +=1
		else:
			write_variant_type(p_id, num_breakpoints,paren_struc,bp_list,seq)
			v_count +=1
	print("Simulated "+str(num_patients) + " patients with " + str(wt_count) +" wild type patients and " + str(v_count) + " variant patients ")
	for i in range(3):
		print("\n")
	print(np.array([q,e,r])/num_patients)
	print(np.array([q,e,r]))
	print("Randomly Displaying Simulated Info for 3 patients")
	id_nums = np.random.choice(np.arange(num_patients),3,replace=False)
	for pid in id_nums:
		fname = PATIENT_SEQS+"patient_"+str(pid)+".txt"
		fstream = open(fname,"r")
		pid = fstream.readline()
		n_locs = fstream.readline()
		parens= fstream.readline()
		bps = fstream.readline()
		fstream.close()
		print_patient(pid,n_locs,parens,bps)



if __name__ == '__main__':
	main()