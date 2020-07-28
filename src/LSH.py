import numpy as np 
import sys





def generate_hash_func(w,k):
	if k>=w:
		print('error')
		sys.exit(0)
	indices = np.arange(1,w+1)
	func_dex = np.random.choice(indices,k,replace=True)
	func = lambda s: ''.join( [ s[i] for i in fun_dex])
	return func



def main():
	test = generate_hash_func(5,3)
	print(test)
