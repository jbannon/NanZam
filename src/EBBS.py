import numpy as np 
import nanzam_utils as utils
import math
import sys
from functools import partial
from collections import Counter
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt


"""
Empirical Bayes Beam Search
	IN:
		seq:=
		HashTable:=
		lenmap:=
		merLen :=
			default=5
		k :=
			default=2
		eps:= 
			default = 0.1
	OUT:
		A parenthentic structure
		A list of candidate sequences
"""

def print_parens(paren_list):
	for struct in paren_list:
		if struct[0]=="(":
			print("(_"+struct[1]+" ",end="")
		elif struct[0]==")":
			print(struct[1]+"_) ",end="")
		else:
			print("(_"+struct[1]+" "+struct[1]+"_) ",end="")
	print("\n")

def compute_score(from_state,to_state,score_val,lN=0,lO=0,e=.9):
	#TODO: change to spec, -log epsilon
	# needs bin map
	w = [0.1,0.9,0.01]
	r=1
	
	s = from_state[1]==to_state[1]-1
	l=from_state[0]==to_state[0]
	t = lN>lO

	#o = from_state[1]
	
	for x,y in zip([s,l,t],w):
		if x:
			r=r*1;
		else:
			r=r*y;
	return score_val*r;

def make_paren(kind, label):
	kind = kind.lower()
	if kind not in ["open", "close", "complete"]:
		print("invalid paren type")
		sys.exit(0)
	if kind=="open":
		return ("(",label)
	elif kind=="close":
		return (")",label)
	else:
		return ("()",label)

def choose_start_state(candidate_list):
	labs = [x[0] for x in candidate_list]
	c = Counter(labs)
	min_lab = c.most_common(1)[0][0]
	minval = float("inf")
	start = candidate_list[0]
	for cand in candidate_list:
		if cand[0]==min_lab and cand[1]<start[1]:
			start = cand
	return start

def resolve_empty(HashTable,m):
	dex =0# utils.hash(m, len(HashTable.keys())) # walk along hash table
	while len(HashTable[dex])==0:
		dex = dex+1
		dex = dex%len(HashTable.keys())
	return HashTable[dex]

def handle_stack(stack,state,count=0):
	# two stacks!
	"""print("STACK IS:",end="")
	print(stack)
	print("STATE IS:",end="")
	print(state)"""
	
	if len(stack)==0:
		"""passed empty stack"""
		stack.append(make_paren("open",state[0]))
		return stack
	T = []
	while len(stack)!=0:
		curr = stack.pop()
		if not(curr[1]==state[0] and curr[0]=="("):
			T.append(curr)
		else:
			stack.append(curr)
			while len(T)!=0:
				stack.append(T.pop())
			if stack[-1][0]=="(" and len(stack)>1 and stack[-1][1]!=state[0]:
				temp = stack.pop()
				new_temp = make_paren("complete",temp[1])
				stack.append(new_temp)
			return stack
	new_paren=make_paren("open",state[0])
	while len(T)!=0:
		stack.append(T.pop())
	#if len(stack)>4:
		#print(stack)
		#sys.exit(0)
	if stack[-1][0]=="(" and len(stack)>1 and stack[-1][1]!=state[0]:
		temp = stack.pop()
		new_temp = make_paren("complete",temp[1])
		stack.append(new_temp)
	stack.append(new_paren)
	return stack
	
def post_process_candidates(candidates):
	for candidate in candidates:
		temp = [x for x in candidate[2]]
		t=make_paren("close",candidate[1])
		last = temp[-1]
		if not last[1]==candidate[1]:
			z = temp.pop()
			temp.append(make_paren("complete",z[1]))
		temp.append(t)
		candidate[2]=temp
	"""TODO add closing of danglers"""
	return candidates

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
	scores = scores[scores<=1e10]
	scores = scores[scores>=-1e10]
	l = np.amin(scores)-.5
	u = np.amax(scores)+0.5
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
	scores = np.array([x[3] for x in candidates])
	scores = scores + np.random.normal(size=scores.shape[0])
	
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

	dex = np.where(np.array(fdrs)<thresh, 1,0)
	if sum(dex)==0:
		ordered = sorted(candidates, key=lambda tup:-tup[-1])
		return ordered[:w]
	elif sum(dex)>=w:
		return candidates[:w]
	else:
		""" change to union??"""
		return candidates[:w]


def EBBS(seq, HashTable, w=5, merLen=5,
	eps=.01,fdr_counter=np.float('inf'),fdr_thresh = 0.2,D=256):
	
	mer0 = "".join(seq[0:merLen])
	score_sets = []


	""" handle w being an integer or a function"""
	""" w, if a function, must be defined on natural numbers (including 0)"""
	if not callable(w):
		w = math.ceil(w)
		temp = w
		w= lambda x: temp

	""" hash the left most mer to get a list of candidate start states from the hash table"""
	start_states = HashTable[utils.hash(mer0, D,len(HashTable.keys()))]
	print(start_states)
	if len(start_states)==0:
		start_states = resolve_empty(HashTable,mer0)
	s_0 = choose_start_state(start_states)
	candidates =[]
	candidates.append([[s_0],s_0[0], [make_paren("open",s_0[0])],1.0])
	count = 0
	for i in range(1,len(seq)-merLen):
		#print("round = "+str(i))
		all_candidates = list()
		mer = seq[i:i+merLen]
		successor_states = HashTable[utils.hash(mer,D,len(HashTable.keys()))]
		if len(successor_states)==0:
			successor_states = resolve_empty(HashTable,mer)
		for l in range(len(candidates)):
			sequence, root, parens, score = candidates[l]
			pl = len(parens)
			for next in successor_states:
				#print("Calling Handle Stack")
				tempP=[x for x in parens]
				temp_list = handle_stack(tempP, next,count)
				new_parens = [x for x in temp_list]
				npl =len(new_parens)
				candidate = [sequence+[next],root, new_parens,compute_score(sequence[-1],next,score, npl, pl)]
				all_candidates.append(candidate)
		if (i+1) % fdr_counter ==0:
			print("doing fdr")

			candidates = fdr_sort(all_candidates,w(count),thresh=0.2)
		else:
			ordered = sorted(all_candidates, key=lambda tup:-tup[-1])
			candidates = ordered[:w(i)]
	candidates=post_process_candidates(candidates)
	return candidates

def main():
	print("Unit Test of EBBS")
	"""
	STACK UNIT TESTS"""

	"""
	
	
	TEST_CASES = [ ["A","B","A"],["A","B","B","A"],["A","B","B","A","A","B","B","A"],
	["A","A","B","A"],["A","B","B","C","C","A"], ["A","B","B","C","C","A"],
	["A","A","B","B","A","C","C","A"],["A","B","A","C","A"], ["A","B","B","B","A","A","A","C","C","A","A"],
	["B","B","A","C","B"],["A","B","C","C","D","B","A"],["A","B","C","D","C","B","A"]]
	stack = []
	#stack=[]
	new_states = ["A","B","B","A",]#"A", "A","B","B","A"]#,"A","A","B","B","A"]
	new_states=["A","B","B","A","A","B","B","A"]
	count = 0
	for state in new_states:
		stack = handle_stack(stack,state)
		#print_parens(stack)
		count+=1
	stack.append(make_paren("close",stack[0][1]))
	
	print("final stack:")
	print_parens(stack)
	#sys.exit(0)
	
	print("Case studies")
	for case in TEST_CASES:
		stack=[]
		print("\n Case:")
		print(case)
		for state in case:
			stack=handle_stack(stack,state)
		stack.append(make_paren("close",stack[0][1]))
		print("final stack:\t",end="")
		print_parens(stack)
	"""
	
	""" FDR Sort Unit Tests"""
	states = [('chr8', 47, 'CBCBB'), ('igh', 1, 'FBBGB'), ('igh', 2, 'BBGBB'), ('igh', 3, 'BGBBB'), ('igh', 4, 'GBBBB'), ('igh', 5, 'BBBBC'), ('igh', 6, 'BBBCB'), ('igh', 7, 'BBCBE'), ('igh', 8, 'BCBEC'), ('igh', 9, 'CBECB'), ('igh', 10, 'BECBB'), ('igh', 11, 'ECBBC'), ('igh', 12, 'CBBCB'), ('igh', 13, 'BBCBD'), ('igh', 14, 'BCBDB'), ('igh', 15, 'CBDBB'), ('igh', 16, 'BDBBB'), ('igh', 17, 'DBBBC'), ('igh', 18, 'BBBCB'), ('igh', 19, 'BBCBB'), ('igh', 20, 'BCBBB'), ('igh', 21, 'CBBBB'), ('igh', 22, 'BBBBB'), ('igh', 23, 'BBBBD'), ('igh', 24, 'BBBDB'), ('igh', 25, 'BBDBC'), ('igh', 26, 'BDBCB'), ('igh', 27, 'DBCBC'), ('igh', 28, 'BCBCB')]
	stack =[]
	for state in states:
		stack = handle_stack(stack,state)
	stack =handle_stack(stack,make_paren("close",stack[0][1]))
	
	print("final stack:")
	print_parens(stack)


	states = [('chr8', 47, 'CBCBB'), ('igh', 1, 'FBBGB'), ('igh', 2, 'BBGBB'), ('igh', 3, 'BGBBB'), ('igh', 4, 'GBBBB'), ('igh', 5, 'BBBBC'), ('igh', 6, 'BBBCB'), ('igh', 7, 'BBCBE'), ('igh', 8, 'BCBEC'), ('igh', 9, 'CBECB'), ('igh', 10, 'BECBB'), ('igh', 11, 'ECBBC'), ('igh', 12, 'CBBCB'), ('igh', 13, 'BBCBD'), ('igh', 14, 'BCBDB'), ('igh', 15, 'CBDBB'), ('igh', 16, 'BDBBB'), ('igh', 17, 'DBBBC'), ('igh', 18, 'BBBCB'), ('igh', 19, 'BBCBB'), ('igh', 20, 'BCBBB'), ('igh', 21, 'CBBBB'), ('igh', 22, 'BBBBB'), ('igh', 23, 'BBBBD'), ('igh', 24, 'BBBDB'), ('igh', 25, 'BBDBC'), ('igh', 26, 'BDBCB'), ('igh', 27, 'DBCBC'), ('igh', 28, 'BCBCB')]
	#sys.exit(0)
	np.random.seed(0)
	x = np.random.randint(-100,100,50)
	x = (x - np.mean(x))/np.std(x)
	l = np.amin(x)#-4.5
	u = np.amax(x)#4.5
	bins = np.arange(l,u,.1)
	h = np.histogram(x,bins)
	
	
	test_cans = [[[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), 
	('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh'), ('(', 'igh')], 3.429999999999999e-05], 
		[[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', 
		[('(', 'chr8'), ('(', 'igh')], 7e-06], [[('chr8', 0, 'BBBBB'),
		 ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-06], 
		 	[[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-06], 
		 		[[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000002e-06]]
	test_cans= [[[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 0, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 1, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 2, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 4, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 5, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 6, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 7, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 8, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 9, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 10, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 11, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 12, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 13, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 14, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 15, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 16, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 17, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 18, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 19, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 20, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 21, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-06], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 25, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 26, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 27, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 28, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 29, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 30, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 31, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 32, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 33, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 34, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 35, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 36, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 37, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 38, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 39, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 40, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 41, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 42, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 55, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 56, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 57, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 58, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 72, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 73, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 74, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 116, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 117, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 118, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 119, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 120, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 121, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 122, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 123, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 132, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 133, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 134, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.9e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('(', 'igh')], 7e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 0, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 1, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 2, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 4, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 5, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 6, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 7, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 8, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 9, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 10, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 11, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 12, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 13, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 14, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 15, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 16, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 17, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 18, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 19, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 20, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 21, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-06], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 25, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 26, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 27, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 28, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 29, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 30, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 31, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 32, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 33, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 34, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 35, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 36, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 37, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 38, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 39, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 40, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 41, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 42, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 55, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 56, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 57, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 58, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 72, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 73, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 74, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 116, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 117, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 118, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 119, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 120, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 121, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 122, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 123, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 132, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 133, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('chr8', 134, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh'), ('(', 'igh')], 3.429999999999999e-05], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 0, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 1, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 2, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000002e-06], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 4, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 5, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 6, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 7, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 8, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 9, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 10, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 11, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 12, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 13, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 14, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 15, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 16, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 17, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 18, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 19, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 20, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 21, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 22, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 25, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 26, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 27, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 28, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 29, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 30, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 31, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 32, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 33, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 34, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 35, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 36, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 37, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 38, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 39, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 40, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 41, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 42, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 55, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 56, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 57, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 58, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 72, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 73, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 74, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 116, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 117, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 118, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 119, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 120, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 121, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 122, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 123, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 132, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 133, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('chr8', 134, 'BBBBB')], 'chr8', [('(', 'chr8')], 1.0000000000000001e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 1, 'BBBBB'), ('chr8', 2, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('(', 'igh')], 7e-06], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 0, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 1, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 2, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 4, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 5, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 6, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 7, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 8, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 9, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 10, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 11, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 12, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 13, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 14, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 15, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 16, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 17, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 18, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 19, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 20, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 21, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 25, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 26, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 27, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 28, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 29, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 30, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 31, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 32, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 33, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 34, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 35, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 36, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 37, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 38, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 39, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 40, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 41, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 42, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 55, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 56, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 57, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 58, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 72, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 73, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 74, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 116, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 117, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 118, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 119, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 120, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 121, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 122, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 123, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 132, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 133, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 134, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('(', 'igh')], 6.999999999999999e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 0, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 1, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 2, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 3, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 4, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 5, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 6, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 7, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 8, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 9, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 10, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 11, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 12, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 13, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 14, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 15, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 16, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 17, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 18, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 19, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 20, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 21, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 23, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.899999999999999e-07], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 24, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 25, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 26, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 27, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 28, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 29, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 30, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 31, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 32, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 33, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 34, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 35, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 36, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 37, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 38, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 39, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 40, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 41, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 42, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 55, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 56, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 57, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 58, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 72, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 73, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 74, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 116, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 117, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 118, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 119, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 120, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 121, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 122, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 123, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 132, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 133, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('chr8', 134, 'BBBBB')], 'chr8', [('(', 'chr8'), ('()', 'igh')], 4.8999999999999995e-08], [[('chr8', 0, 'BBBBB'), ('chr8', 0, 'BBBBB'), ('igh', 22, 'BBBBB'), ('igh', 22, 'BBBBB')], 'chr8', [('(', 'chr8'), ('(', 'igh')], 6.999999999999999e-08]]	 			
	fdr_sort(test_cans,5)
	sys.exit(0)
	y=h[0]
	x = h[1]
	print(y)
	print(x)
	sys.exit(0)
	midpoints = (x[1:]+x[:-1])/2
	
	f_0 = lambda x: norm.pdf(x,0,1); ## theoretical null distribution
	J = 7
	res = np.zeros((midpoints.shape[0],J+1))

	powers = np.arange(0,J+1)
	for i in range(midpoints.shape[0]):
		res[i]=midpoints[i]**powers
	#res =sm.tools.tools.add_constant(res)
	model = sm.GLM(y, res, family=sm.families.Poisson())
	model =model.fit()
	print(model.summary())
	#print(model.params)
	#print(len(model.params))
	pi_0=0.9
	#sys.exit(0)
	candidates = []
	for d in x:
		 candidates.append([ [('chr8', 47, 'CBCBB'), ('igh', 1, 'FBBGB')], 'chr8', [('(', 'chr8'), ('(', 'igh')], d])

	
	m = x
	preds = np.zeros(len(m))
	featurized_m = np.zeros((m.shape[0],J+1))
	for i in range(m.shape[0]):
		featurized_m[i]=m[i]**powers
	for j in range(len(m)):
		preds[j] = get_fdr(m[j],pi_0,f_0,model)
		if preds[j]<0:
			print("neg")
		
	#print(candidates[0])
	print(candidates[0][3])
	print(candidates[0])
	c1=sorted(candidates, key=lambda cand: get_fdr(cand[3],pi_0,f_0,model)[0])
	c2,d1 = fdr_sort(candidates,model,f_0, pi_0)
	print(c1)
	print(c2)
	#print(get_fdr(np.float64(1.4671559003586507),pi_0,f_0,model))
	for z in c1:
		print(get_fdr(np.float64(z[3]),pi_0,f_0,model))
	print(d1)
	#plt.plot(m,preds)
	#plt.show()


if __name__ == '__main__':
	main()