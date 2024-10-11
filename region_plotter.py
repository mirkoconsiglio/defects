import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

d = np.linspace(1, 30, 1001)
eps_1_2 = -1 / d
eps_2_3 = -3 / d

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel(r'$d$')
ax.set_ylabel(r'$\epsilon$')
ax.fill_between(d, 0, eps_1_2, alpha=0.5, label='1 defect')
ax.plot(d, eps_1_2, label=r'$\epsilon = -\frac{1}{d}$')
ax.fill_between(d, eps_1_2, eps_2_3, alpha=0.5, label='2 defects')
ax.plot(d, eps_2_3, label=r'$\epsilon = -\frac{3}{d}$')
ax.fill_between(d, eps_2_3, -3, alpha=0.5, label='3 defects')
ax.legend(loc='lower right')

fig.savefig(f'figures/plot.pdf', bbox_inches='tight')

plt.show()
