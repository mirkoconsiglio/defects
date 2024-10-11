import json

import matplotlib.pyplot as plt
from matplotlib import rc
import colorcet as cc
import numpy as np

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

_colour_map = cc.cm.CET_C8s
_colour_list = np.transpose(list(_colour_map._segmentdata.values()))[1]


def plot_eps(data):
	h = data.get('h')
	eps = data.get('eps')
	purity = data.get('purity')
	conc = data.get('conc')
	tri_neg = data.get('tri_neg')
	
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\epsilon$')
	ax.set_ylabel(r'$E$')
	ax.grid(visible=True, which='both', axis='both')
	
	markers = ['o', 's', 'd', '^', 'X']
	colours = _colour_list[0::28]
	
	# ax.plot(eps, purity, color=colours[0], linestyle='-', label='Purity')
	ax.plot(eps, conc, marker=markers[1], color=colours[1], linestyle='-', label='Concurrence')
	ax.plot(eps, tri_neg, marker=markers[2], color=colours[2], linestyle='-', label='Tripartite Negativity')
	
	ax.legend(title=f'$h = {h}$')
	
	fig.savefig(f'figures/plot.pdf', bbox_inches='tight')
	fig.savefig(f'figures/plot.png', dpi=600, transparent=True, bbox_inches='tight')
	
	plt.show()


def plot_h(data):
	h = data.get('h')
	eps = data.get('eps')
	purity = data.get('purity')
	conc = data.get('conc')
	tri_neg = data.get('tri_neg')
	
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$h$')
	ax.set_ylabel(r'$E$')
	ax.grid(visible=True, which='both', axis='both')
	
	markers = ['o', 's', 'd', '^', 'X']
	colours = _colour_list[0::28]
	
	# ax.plot(h, purity, color=colours[0], linestyle='-', label='Purity')
	ax.plot(h, conc, marker=markers[1], color=colours[1], linestyle='-', label='Concurrence')
	ax.plot(h, tri_neg, marker=markers[2], color=colours[2], linestyle='-', label='Tripartite Negativity')
	
	ax.legend(title=fr'$\epsilon = {eps}$')
	
	fig.savefig(f'figures/plot_2.pdf', bbox_inches='tight')
	fig.savefig(f'figures/plot_2.png', dpi=600, transparent=True, bbox_inches='tight')
	
	plt.show()


def main():
	with open('data.json') as f:
		data = json.load(f)
	
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$-\epsilon d$')
	ax.set_ylabel(r'$C_{\mathrm{GME}}$')
	ax.grid(visible=True, which='both', axis='both')
	for x in data:
		eps = np.array(x[0])
		conc = np.array(x[1])
		d = x[2]
		ax.plot(eps * d, conc, label=fr'$d={d}$')
	ax.legend()
	plt.savefig('gme.pdf')
	plt.show()


if __name__ == '__main__':
	main()
