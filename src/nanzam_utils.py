import pickle
import string
import sys
import numpy as np
from constants import *

def makeTable(table_size):
	"""
	speeds up so we don't have if checks
	"""
	table = {}
	for k in range(table_size):
		table[k] = []
	return table


def hash(merString,D=256,q=101):
	"""
	computes rabin karp fingerprint of the string

	"""
	hashval = 0
	for i in range(len(merString)): # preprocessing
		hashval = (hashval*D+ord(merString[i]))%q
	return hashval
	

def make_almap(bins):
	if len(bins)<1:
		print("error, length of bins less than one, aborting")
		sys.exit(0)
	elif len(bins)<=9:
		alphabet = SIMPLE_ALPHS[len(bins)-1]
	else:
		lowers = list(string.ascii_lowercase)
		uppers = list(string.ascii_uppercase)
		uppers.extend(lowers)
		alphabet = uppers[0:len(bins)]
	almap = {}
	for dex in range(len(bins)-1):
		range_tup = (bins[dex],bins[dex+1])
		almap[range_tup]=alphabet[dex]
	return alphabet, almap
		

def make_Alphabet(length_array, n_bins,add_two=True):
	if add_two:
		""" add two"""
		counts,bins = np.histogram(length_array,n_bins)
		bins = list(bins)
		bins.insert(0,float("-inf"))
		bins.append(float("inf"))
		alphabet, almap = make_almap(bins)
		return alphabet,almap,bins
	else:
		"""keep bin static"""
		counts,bins = np.histogram(length_array,n_bins)
		bins = list(bins)[1:len(bins)-1]
		bins.insert(0,float("-inf"))
		bins.append(float("inf"))
		#print(bins)
		alphabet, almap = make_almap(bins)
		return alphabet,almap,bins


def boundary_query(y,range_tuple,eps):
	member,upper_close,lower_close=False,False,False
	if range_tuple[0]<=y and y<range_tuple[1]:
		member=True
		if np.abs(y-range_tuple[0])<eps:
			lower_close=True
		elif np.abs(range_tuple[1]-y)<eps:
			upper_close=True
	return member, upper_close,lower_close


def boundary_translate(y,almap,eps):
	res_list =[]
	for dex in range(len(almap.keys())):
		I=list(almap.keys())[dex]
		m,u,l =boundary_query(y,I,eps)
		if m:
			res_list.append(almap[I])
			if u:
				res_list.append(almap[list(almap.keys())[dex+1]])
			elif l:
				res_list.append(almap[list(almap.keys())[dex-1]])
			return res_list


def translate2(length_vector,almap,jitter=True,epsilon=0.01):
	if jitter:
		possible_maps=[]
		for j in range(len(length_vector)):
			new_letters=[x for x in boundary_translate(length_vector[j],almap,epsilon)]
			if len(possible_maps)==0:
				for letter in new_letters:
					possible_maps.append([letter])
			else:
				new_maps=[]
				for possible_map in possible_maps:
					for letter in new_letters:
						temp = [x for x in possible_map]
						temp.append(letter)
						new_maps.append(temp)
				possible_maps=new_maps
		return possible_maps
	else:
		#translate from old code base
		translated_vector = []
		for j in range(len(length_vector)):
			translated_vector.append(range_translate(length_vector[j],almap))
		return translated_vector

def range_query(y,range_tuple):
	if range_tuple[0]<=y and y<range_tuple[1]:
		return True
	else:
		return False


def range_translate(y,almap):
	for I in almap.keys():
		if range_query(y,I):
			return almap[I]




def main():
	print("make Alphabet unit test")
	lengths = [1,2,3,4,5,6,6,7,9]
	n_bins = 3
	a,b,c = make_Alphabet(lengths, n_bins,True)
	print(b)
	eps =0.01
	x=[-0.001,.5,1.5,2.001,10]
	z1 =translate(x,b)
	b = { (np.float('-inf'),0 ): 'A', (0,1): 'B', (1,2): 'C', (2,np.float('inf')): 'D'}
	z2=translate2(x,b)
	print(z1)
	print(z2)




if __name__ == '__main__':
	main()


