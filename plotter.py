import json
import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from utils import gme, concurrences

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 26})
rc('text', usetex=True)

_colour_map = cc.cm.CET_L8
_colour_list = np.transpose(list(_colour_map._segmentdata.values()))[1]
colours = _colour_list[0::26]


def region_plotter():
	d = np.linspace(1, 10, 1001)
	eps_1_2 = -1 / d
	eps_2_3 = -3 / d
	
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$d$')
	ax.set_ylabel(r'$\epsilon$')
	ax.fill_between(d, 0, eps_1_2, color=colours[0], alpha=0.5, label='1 Localized State')
	ax.plot(d, eps_1_2, color=colours[2], label=r'$\varepsilon = \frac{1}{d}$')
	ax.fill_between(d, eps_1_2, eps_2_3, color=colours[4], alpha=0.5, label='2 Localized States')
	ax.plot(d, eps_2_3, color=colours[6], label=r'$\varepsilon = \frac{3}{d}$')
	ax.fill_between(d, eps_2_3, -3.5, color=colours[8], alpha=0.5, label='3 Localized States')
	ax.legend(loc='lower right', fontsize='18')
	
	fig.savefig(f'defects.pdf', bbox_inches='tight')


# plt.show()


def conc_plot(directory):
	fig, axes = plt.subplots(1, 2, sharey=True, figsize=(30, 10))
	axes[0].set_ylabel(r'$\mathcal{C}$')
	axes[1].tick_params(left=False, labelleft=False)
	fig.subplots_adjust(wspace=0)
	for ax in axes:
		ax.set_xlabel(r'$\varepsilon d$')
		ax.grid(visible=True, which='both', axis='both')
		ax.axvline(1, color='k', linestyle='--')
		ax.axvline(3, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir(directory)):
		if file.endswith('.json'):
			i += 1
			file_path = os.path.join(directory, file)
			
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data]
			conc_list = np.array([concurrences(i['matrix']) for i in data])
			
			axes[0].plot(eps_d_list, conc_list[:, 0], color=colours[i], marker='.', markersize=10,
			         label=fr'$d={data[0]['d']}$')
			axes[1].plot(eps_d_list, conc_list[:, 1], color=colours[i], marker='.', markersize=10,
			         label=fr'$d={data[0]['d']}$')
	
	axes[0].legend(title=r'$\mathcal{C}_{12}$')
	axes[1].legend(title=r'$\mathcal{C}_{13}$')
	plt.savefig(f'conc_{directory}.pdf')


# plt.show()


def gme_plot(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\varepsilon d$')
	ax.set_ylabel(r'$\mathcal{C}_{\mathrm{GME}}$')
	ax.grid(visible=True, which='both', axis='both')
	ax.axvline(1, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir(directory)):
		if file.endswith('.json'):
			i += 1
			file_path = os.path.join(directory, file)
			
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data if i['eps_d'] <= 1]
			gme_list = [gme(i['matrix']) for i in data][:len(eps_d_list)]
			
			ax.plot(eps_d_list, gme_list, color=colours[i], marker='.', markersize=10, label=fr'$d={data[0]['d']}$')
	
	ax.legend(fontsize='18')
	plt.savefig(f'gme_{directory}.pdf')


# 	plt.show()


def hong_plot(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\varepsilon d$')
	ax.set_ylabel(r'$\mathcal{C}_{\mathrm{GME}}^{\mathrm{Hong}}$')
	ax.grid(visible=True, which='both', axis='both')
	ax.axvline(1, color='k', linestyle='--')
	ax.axvline(3, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir(directory)):
		if file.endswith('.json'):
			i += 1
			file_path = os.path.join(directory, file)
			
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data]
			chen_list = [np.sqrt(2) * i['chen_measure'] for i in data]
			
			ax.plot(eps_d_list, chen_list, color=colours[i], marker='.', markersize=10, label=fr'$d={data[0]['d']}$')
	
	ax.legend(fontsize='18')
	plt.savefig(f'chen_{directory}.pdf')


# 	plt.show()


def ma_plot(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\varepsilon d$')
	ax.set_ylabel(r'$\mathcal{C}_{\mathrm{GME}}^{\mathrm{Ma}}$')
	ax.grid(visible=True, which='both', axis='both')
	ax.axvline(1, color='k', linestyle='--')
	ax.axvline(3, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir(directory)):
		if file.endswith('.json'):
			i += 1
			file_path = os.path.join(directory, file)
			
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data]
			huber_list = [i['huber_measure'] for i in data]
			
			ax.plot(eps_d_list, huber_list, color=colours[i], marker='.', markersize=10, label=fr'$d={data[0]['d']}$')
	
	ax.legend(fontsize='18')
	plt.savefig(f'huber_{directory}.pdf')


# 	plt.show()


def both_plot(directory):
	fig, axes = plt.subplots(1, 2, sharey=True, figsize=(30, 10))
	axes[0].set_ylabel(r'$\mathcal{C}_{\mathrm{GME}}^\mathrm{lb}$')
	axes[1].tick_params(left=False, labelleft=False)
	fig.subplots_adjust(wspace=0)
	for ax in axes:
		ax.set_xlabel(r'$\varepsilon d$')
		ax.grid(visible=True, which='both', axis='both')
		ax.axvline(1, color='k', linestyle='--')
		ax.axvline(3, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir(directory)):
		if file.endswith('.json'):
			i += 1
			file_path = os.path.join(directory, file)
			
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data]
			huber_list = [i['huber_measure'] for i in data]
			chen_list = [np.sqrt(2) * i['chen_measure'] for i in data]
			
			axes[0].plot(eps_d_list, huber_list, color=colours[i], marker='.', markersize=10,
			             label=fr'$d={data[0]['d']}$')
			axes[1].plot(eps_d_list, chen_list, color=colours[i], marker='.', markersize=10,
			             label=fr'$d={data[0]['d']}$')
	
	axes[0].legend(title=r'$\mathcal{C}_{\mathrm{GME}}^\mathrm{Ma}$')
	axes[1].legend(title=r'$\mathcal{C}_{\mathrm{GME}}^\mathrm{Hong}$')
	plt.savefig(f'both_{directory}.pdf')


# plt.show()


def localized_plot(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\varepsilon d$')
	ax.set_ylabel(r'$\omega_k$')
	ax.grid(visible=True, which='both', axis='both')
	
	with open('localised_d_1.json', 'r') as json_file:
		data = json.load(json_file)
	
	eps_d_list = np.linspace(0, 5, len(data))
	eigenvalues = np.array(data)[:, -3:]
	
	for i, j in enumerate(np.transpose(eigenvalues)[::-1]):
		ax.plot(eps_d_list, j, color=colours[3 * i], markersize=10, label=fr'$\omega_{i + 1}$')
	
	ax.fill_between(eps_d_list, 0, 4, color=colours[8], alpha=0.5, label='Band')
	
	ax.axvline(1, color='k', linestyle='--')
	ax.axvline(3, color='k', linestyle='--')
	
	props = dict(boxstyle='round', alpha=0.5, color=colours[9])
	ax.text(0.04, 0.07, '1 Localized State', transform=ax.transAxes, fontsize=16,
	        verticalalignment='top', bbox=props)
	ax.text(0.32, 0.07, '2 Localized States', transform=ax.transAxes, fontsize=16,
	        verticalalignment='top', bbox=props)
	ax.text(0.68, 0.07, '3 Localized States', transform=ax.transAxes, fontsize=16,
	        verticalalignment='top', bbox=props)
	
	ax.legend(fontsize=18)
	plt.savefig(f'localised_{directory}.pdf')


# plt.show()

def test_plot():
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\varepsilon d$')
	ax.set_ylabel(r'$y$')
	ax.grid(visible=True, which='both', axis='both')
	ax.axvline(1, color='k', linestyle='--')
	ax.axvline(3, color='k', linestyle='--')
	
	i = 0
	for file in np.sort(os.listdir('test')):
		if file.endswith('d_5.json'):
			i += 1
			file_path = os.path.join('test', file)
			with open(file_path, 'r') as json_file:
				data = json.load(json_file)
			
			eps_d_list = [i['eps_d'] for i in data]
			y_list = [i['u'] for i in data]
			
			ax.plot(eps_d_list, y_list, marker='.', markersize=10)
	
	ax.legend(fontsize='18')
	plt.show()


if __name__ == '__main__':
	# test_plot()
	# quit()
	
	region_plotter()
	for directory in ['h_2.0', 'h_1.0', 'h_0.0']:
		if directory == 'h_2.0':
			gme_plot(directory)
			localized_plot(directory)
		both_plot(directory)
		conc_plot(directory)
		ma_plot(directory)
		hong_plot(directory)
