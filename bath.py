import h5py 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import numpy as np
from matplotlib.ticker import FuncFormatter


class BATH_FITTING:
	def __init__(self, file_path, Hamil, Bnum):
		self.H = Hamil;
		self.N = Bnum;
		self.File = file_path;	
		self.gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[6], width_ratios=[12, 1])
		self.snum, self.bath, self.chi = self.call_data()

	def format_func(self, value, tick_number):
		return f'{value:.2e}';
	
	def call_data(self):
		snum = 10;
		bath = [];	chi = [];
		with h5py.File(self.File, 'r') as file:
			bath = [file[f'{self.H}/N{self.N}/P0']];
			chi_arr = bath[0].attrs['chi'];
			chi.append(chi_arr[0]);
			for i in range(1,snum):
				add_data = file[f'{self.H}/N{self.N}/P{i}'];
				chi_arr = np.vstack( (chi_arr, add_data.attrs['chi']) );
				bath = np.vstack( (bath, [add_data]) );
				chi.append(chi_arr[i][0]);

		return snum, bath, chi;

	def plot_bath(self):
		snum, bath = self.snum, self.bath;
		axis = plt.subplot(self.gs[0]);	
		Nb = self.N;	
		colors = ['blue', 'red'];
		labels = ['Initial', 'Final'];
		for i in range(snum):
			y_ticks = np.full(Nb, i);
			for j in range(2):
				axis.scatter(bath[i][j][Nb:], y_ticks+.2*j, color=colors[j], marker='o', s=20, label=labels[j] if i==0 else None);
				for k in range(Nb):
					axis.plot([bath[i][j][Nb+k]-bath[i][j][k]/2, bath[i][j][Nb+k]+bath[i][j][k]/2], [i+.2*j,i+.2*j], color=colors[j], lw=3);
					axis.scatter([bath[i][j][Nb+k]-bath[i][j][k]/2, bath[i][j][Nb+k]+bath[i][j][k]/2], [i+.2*j,i+.2*j], color=colors[j], marker='|', s=50);
		
		axis.set_yticks([]);
		axis.set_xlabel('$\epsilon$', fontsize=40);
		axis.set_ylabel('Index of initial conditions', fontsize=30);
		axis.legend(bbox_to_anchor=(1., 1), loc='upper left', fontsize=20)
#		legend.get_frame().set_alpha(.4); #transparent 0~1

	def plot_chi(self): 
		snum, chi = self.snum, self.chi;	
		chi = self.call_data()[2];
		ax = plt.subplot(self.gs[1], sharey=plt.subplot(self.gs[0]) );	
		for i in range(snum):
			ax.axvline(chi[i], color='gray', linestyle=':');
		ax.scatter(chi, np.arange(.1, 9.2, 1), color='green', marker='o', s=20);

		ax.xaxis.set_major_formatter(FuncFormatter(self.format_func));
		ax.set_xticks([chi[0], chi[-1]])
		plt.tick_params(axis='x', length=2, width=2, labelsize=15)
		ax.set_xlabel('$\chi^2$', fontsize=25);	
		ax.xaxis.set_label_coords(.4, -.033);
		
	
	def generate_plot(self):
		fig = plt.figure(figsize=(21,9));
		self.plot_chi();
		self.plot_bath(); 
		plt.suptitle(f'$N_b={N}$', fontsize=30)
		plt.tight_layout()
		plt.savefig(f'N{self.N}.png', dpi=500, transparent=True);
		plt.show();


# os.path.join: connect to path, os.getcwd(): get current working directory
parser = argparse.ArgumentParser();
parser.add_argument('--hamiltonian', '-H', type=str, dest='hamiltonian');
parser.add_argument('--bathnum', '-N', type=str, nargs='+',dest='bathnum');
args = parser.parse_args();

H = args.hamiltonian;	N = int(args.bathnum[0]);

plot_generator = BATH_FITTING('BATH_FIT.h5', H, N);
plot_generator.generate_plot();

