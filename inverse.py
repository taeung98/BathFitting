import os
from matplotlib.ticker import LogLocator
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import numpy as np

def load_data(file_path):
	data = {'x': [], 'y': [], 'z': []}
	with open(file_path, "r") as f:
		data_file = f.read().splitlines()

	for line in data_file:
		parts = line.split()
		data['x'].append(float(parts[0]))
		data['y'].append(float(parts[1]))
		data['z'].append(float(parts[2]))
	return data

def plot_data(folder_path_1,folder_path_2):
	folder_path_i = os.path.join(os.getcwd(),folder_path_1[0][0])
	file_list_i = os.listdir(folder_path_i)
		
	folder_path_f = os.path.join(os.getcwd(),folder_path_2[0][0])
	file_list_f = os.listdir(folder_path_f)
	
	fig = plt.figure(figsize=(21,15))
	gs = gridspec.GridSpec(nrows=1,ncols=1,height_ratios=[6],width_ratios=[6])
	ax1 = plt.subplot(gs[0])
	origin_data = load_data(os.path.join(folder_path_i,file_list_i[0]))
	plt.tick_params(axis='x',length=3,width=3,labelsize=45)
	plt.tick_params(axis='y',length=3,width=3,labelsize=45)
	plt.yticks([-0.25,-0.27,-0.29],('-0.25','-0.27','-0.29'))
	plt.xticks([0,64,128],('0','$\\beta/2$','$\\beta$(=128)'))

	ax1.plot(origin_data['x'][1:-1],origin_data['y'][1:-1],lw=8,color='black',label='Original')
	ax1.set_xlabel('$\\tau$',fontsize=40)
	ax1.set_ylabel('$Re[\Delta_t(\\tau)$]',fontsize=40)
	
	c =0 ; C = []; line = ['solid','dashed','dashdot',(5,(10,3))]

	for txt_file_f in file_list_f:
		C.append(int(txt_file_f[:-4]))
	C.sort();
	for i in C:
		file_path_f = os.path.join(folder_path_f, str(i)+'.txt')
		data_f = load_data(file_path_f);
		ax1.plot(data_f['x'][1:-1],data_f['y'][1:-1],lw=5,ls=line[c],label=f'$N_\ell$={i}')
		c +=1;

	plt.legend(fontsize=35)
	plt.savefig('Dt_re.png',dpi=500,transparent=True)	
	
	plt.show()

	return 0

# os.path.join: connect to path, os.getcwd(): get current working directory
parser = argparse.ArgumentParser()
parser.add_argument('--original', '-o', action='append', nargs='+',dest='original')
parser.add_argument('--inverse', '-i', action='append', nargs='+',dest='inverse')
args = parser.parse_args()
plot_data(args.original, args.inverse)
