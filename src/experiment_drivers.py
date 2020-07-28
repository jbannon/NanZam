import sys
import pickle
import nanzam_utils as utils
import pandas as pd
import numpy as np
import os
import time
import bisect
from EBBS import * 
import plot_utils
from constants import *
import NanZam
from bayes_opt import BayesianOptimization
import plot_utils as pltools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def base_nanzam():
	NanZam.run_offline_phase()
	true_neg, false_pos, false_neg, true_pos, conf_mat = NanZam.run_online_phase()


def plot_multiline(x,ys,xlab,ylab,title,fname,labels,auto_ticks=True):
	colors=['r','g','b','y']
	if auto_ticks:
		for i in range(len(ys)):
			y=ys[i]
			plt.plot(x,y,label=labels[i],linestyle='--',color=colors[i%len(colors)])
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title(title)
		ax = plt.subplot(111)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.legend()
		plt.savefig(fname)
		#plt.show()
		plt.close()
	else:
		for i in range(len(ys)):
			print(colors[i%len(colors)])
			y=ys[i]
			plt.plot(x,y,label=labels[i],linestyle='--',color=colors[i%len(colors)])
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title(title)
		plt.xticks(x)
		ax = plt.subplot(111)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.legend()
		plt.savefig(fname)
		#plt.show()
		plt.close()
def plot_tradeoff(x,y,xlab,ylab,title,fname,auto_ticks=True):
	if auto_ticks:
		plt.plot(x,y)
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title(title)
		ax = plt.subplot(111)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.savefig(fname)
		#plt.show()
		plt.close()
	else:
		plt.plot(x,y)
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title(title)
		plt.xticks(x)
		ax = plt.subplot(111)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.savefig(fname)
		#plt.show()
		plt.close()


def plot_surface(X,Y,Z,xlabel,ylabel,zlabel,title,fname=""):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X,Y = np.meshgrid(X,Y)
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	plt.title(title)
	#plt.show()
	plt.savefig(fname)



def aggregate_score(kappa, fp_rate, fn_rate,thresh = 10**6, minimize_me=True):
	'''
		way of scoring nanzam results such that

		@param kappa: emphasis parameter deciding on how much we care about 
					false negative versus false negatives [0,1]
		@param fp_rate: reported false positive rate
		@param fn_rate: reported false negative rate
		@param minimize_me=True boolean parameter about whether this is a
								function we want to minimize (True) or maximize (False)
	'''
	if minimize_me:
		return kappa*fp_rate + (1-kappa)*fn_rate
	else:
		return  1/min(thresh, kappa*fp_rate + (1-kappa)(fn_rate))
def main():
	""" fixed beam widths """
	bws = [5,10,15,20,200]
	fdrs = [1,10,50,100,200]
	bws = np.arange(1,40)
	#bws = np.arange(1,20)
	
	kappa = 1.0 # parameter to select emphasis of 

	
	tt=[]
	tns,fps, fns, tps,accs =[],[],[],[],[]
	for bw in bws:
		true_neg, false_pos, false_neg, true_pos, conf_mat,time,accuracy= NanZam.run_online_phase(w=bw)
		tt.append(time)
		tns.append(true_neg/19)
		fps.append(false_pos/181)
		fns.append(false_neg/19)
		tps.append(true_pos/181)
		accs.append(accuracy)
	
	labs = ['True Negative','False Positive','False Negative','True Positive']#,'Accuracy']
	ys = [tns, fps, fns, tps]#
	plot_tradeoff(bws,tt,"Beam Width","Average Runtime per Patient (seconds)","NanZam Average Runtime vs. Beam Width","../figs/beamwidth_runtime.pdf",auto_ticks=True)
	plot_multiline(bws,ys,"Beam Width","System Performance","NanZam Performance vs. Beam Width","../figs/beamwidth_performance.pdf",labels=labs,auto_ticks=True)
	plot_tradeoff(bws,accs,"Beam Width","Accuracy","NanZam Accuracy vs. Beam Width","../figs/beamwidth_acc.pdf",auto_ticks=True)
	
	mer_lens = np.arange(2,12)
	alsizes = np.arange(2,27)

	## surface plot for mer lengths
	"""
	SETTINGS 1
	
	fdr_freq=np.float('inf');
	beam_width=5;
	
	#mer_lens = [2,5,8] #5
	#alsizes = [6,8]#8


	tnr=np.zeros((len(mer_lens),len(alsizes)))
	tpr=np.zeros((len(mer_lens),len(alsizes)))
	accs = np.zeros((len(mer_lens),len(alsizes)))
	for i in range(len(mer_lens)):
		for j in range(len(alsizes)):
			NanZam.run_offline_phase(mer_length=mer_lens[i],alphsize=alsizes[j])
			true_neg, false_pos, false_neg, true_pos, conf_mat,time,accuracy= NanZam.run_online_phase(mer_length=mer_lens[i],w=beam_width,fdr_counter=fdr_freq)
			accs[i,j]=accuracy
			tnr[i,j]=true_neg/19
			tpr[i,j]=true_pos/181
	plot_surface(mer_lens,alsizes,np.transpose(accs),"Mer Lengths","Alphabet Size","Accuracy","Accuracy vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_acc_1.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tpr),"Mer Lengths","Alphabet Size","True Positive Rate","True Positive Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tpr_1.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tnr),"Mer Lengths","Alphabet Size","True Negative Rate","True Negative Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tnr_1.pdf")
	
	"""
	"""
	#Settings 2
	fdr_freq=20;
	beam_width=5;

	tnr=np.zeros((len(mer_lens),len(alsizes)))
	tpr=np.zeros((len(mer_lens),len(alsizes)))
	accs = np.zeros((len(mer_lens),len(alsizes)))
	for i in range(len(mer_lens)):
		for j in range(len(alsizes)):
			print('hi')
			NanZam.run_offline_phase(mer_length=mer_lens[i],alphsize=alsizes[j])
			true_neg, false_pos, false_neg, true_pos, conf_mat,time,accuracy= NanZam.run_online_phase(mer_length=mer_lens[i],w=beam_width,fdr_counter=fdr_freq)
			accs[i,j]=accuracy
			tnr[i,j]=true_neg/19
			tpr[i,j]=true_pos/181
	plot_surface(mer_lens,alsizes,np.transpose(accs),"Mer Lengths","Alphabet Size","Accuracy","Accuracy vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_acc_2.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tpr),"Mer Lengths","Alphabet Size","True Positive Rate","True Positive Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tpr_2.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tnr),"Mer Lengths","Alphabet Size","True Negative Rate","True Negative Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tnr_2.pdf")

	# Settings 3
	fdr_freq=20;
	beam_width=lambda t:np.floor(max(15*np.exp(t),3))

	tnr=np.zeros((len(mer_lens),len(alsizes)))
	tpr=np.zeros((len(mer_lens),len(alsizes)))
	accs = np.zeros((len(mer_lens),len(alsizes)))
	for i in range(len(mer_lens)):
		for j in range(len(alsizes)):
			print('hi')
			NanZam.run_offline_phase(mer_length=mer_lens[i],alphsize=alsizes[j])
			true_neg, false_pos, false_neg, true_pos, conf_mat,time,accuracy= NanZam.run_online_phase(mer_length=mer_lens[i],w=beam_width,fdr_counter=fdr_freq)
			accs[i,j]=accuracy
			tnr[i,j]=true_neg/19
			tpr[i,j]=true_pos/181
	plot_surface(mer_lens,alsizes,np.transpose(accs),"Mer Lengths","Alphabet Size","Accuracy","Accuracy vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_acc_3.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tpr),"Mer Lengths","Alphabet Size","True Positive Rate","True Positive Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tpr_3.pdf")
	plot_surface(mer_lens,alsizes,np.transpose(tnr),"Mer Lengths","Alphabet Size","True Negative Rate","True Negative Rate vs. Mer-Length and Alphabet-Size",fname="../figs/meralph_tnr_3.pdf")

	"""	
	tt=[]
	tns,fps, fns, tps,accs =[],[],[],[],[]
	fdrs = [5,10,15,20,25,30,35,40,45,50]
	for fdr in fdrs:
		true_neg, false_pos, false_neg, true_pos, conf_mat,time,accuracy= NanZam.run_online_phase(fdr_counter=fdr)
		tt.append(time)
		tns.append(true_neg/19)
		fps.append(false_pos/181)
		fns.append(false_neg/19)
		tps.append(true_pos/181)
		accs.append(accuracy)
	labs = ['True Negative','False Positive','False Negative','True Positive']#,'Accuracy']
	ys = [tns, fps, fns, tps]#
	plot_tradeoff(fdrs,tt,"FDR Frequency","Average Runtime per Patient (seconds)","NanZam Average Runtime vs. FDR Frequency","../figs/fdrfreq_runtime.pdf",auto_ticks=False)
	plot_multiline(fdrs,ys,"FDR Frequency","System Performance","NanZam Performance vs. FDR Frequency","../figs/fdr_performance.pdf",labels=labs,auto_ticks=False)
	plot_tradeoff(fdrs,accs,"FDR Frequency","Accuracy","NanZam Accuracy vs. FDR Frequency","../figs/fdr_acc.pdf",auto_ticks=False)
	


if __name__ == '__main__':
	main()
	
	