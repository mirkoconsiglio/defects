from concurrent.futures import ProcessPoolExecutor

from qiskit.quantum_info import partial_trace
from scipy.linalg import eigvals
from scipy.sparse import lil_matrix
from scipy.optimize import minimize
import numpy as np


def index_to_qubit_indices(n, num_qubits):
	# Convert (n-1) to binary and pad with zeros to match the number of qubits
	return np.binary_repr(n - 1, width=num_qubits)


def permutation_matrix(permutation):
	num_qubits = len(permutation)
	size = 2 ** num_qubits
	mat = lil_matrix((size, size))  # Sparse matrix of size 2^num_qubits x 2^num_qubits
	
	for i in range(1, size + 1):
		i_bin = list(index_to_qubit_indices(i, num_qubits))  # Convert index to binary list
		i_bin = [int(x) for x in i_bin]  # Convert binary string to list of integers
		permuted_bin = [i_bin[p - 1] for p in permutation]  # Apply the permutation
		j = int(''.join(map(str, permuted_bin)), 2) + 1  # Convert back to integer
		mat[i - 1, j - 1] = 1  # Set the appropriate matrix entry to 1
	
	return mat


def M(i):
	if i == 1:
		return permutation_matrix([4, 2, 3, 1, 5, 6]).todense()
	elif i == 2:
		return permutation_matrix([1, 5, 3, 4, 2, 6]).todense()
	elif i == 3:
		return permutation_matrix([1, 2, 6, 4, 5, 3]).todense()


def apply_kron(psi_list):
	result = psi_list[0]  # Start with the first vector/matrix in the list
	for psi in psi_list[1:]:  # Apply np.kron iteratively for the rest
		result = np.kron(result, psi)
	return result


def U(theta, phi):
	# Define the matrix elements
	element_11 = np.exp(-1j * phi / 2) * np.cos(theta / 2)
	element_12 = -np.exp(-1j * phi / 2) * np.sin(theta / 2)
	element_21 = np.exp(1j * phi / 2) * np.sin(theta / 2)
	element_22 = np.exp(1j * phi / 2) * np.cos(theta / 2)
	
	# Create the matrix
	return np.array([
		[element_11, element_12],
		[element_21, element_22]
	])


def huber_cost(params, rho, M1, M2, M3, M123):
	psi1 = [np.cos(params[0] / 2), np.exp(1j * params[1]) * np.sin(params[0] / 2)]
	psi2 = [np.cos(params[2] / 2), np.exp(1j * params[3]) * np.sin(params[2] / 2)]
	psi3 = [np.cos(params[4] / 2), np.exp(1j * params[5]) * np.sin(params[4] / 2)]
	psi4 = [np.cos(params[6] / 2), np.exp(1j * params[7]) * np.sin(params[6] / 2)]
	psi5 = [np.cos(params[8] / 2), np.exp(1j * params[9]) * np.sin(params[8] / 2)]
	psi6 = [np.cos(params[10] / 2), np.exp(1j * params[11]) * np.sin(params[10] / 2)]
	psi = apply_kron([psi1, psi2, psi3, psi4, psi5, psi6])
	rho2 = np.outer(rho, rho)
	
	f = np.real(np.sqrt(psi.conj() @ rho2 @ M123 @ psi) - np.sum([
		np.sqrt(psi.conj() @ M.T @ rho2 @ M @ psi) for M in [M1, M2, M3]]))[0, 0]
	
	return -f


def minimize_huber_cost(rho, M1, M2, M3, M123):
	x0 = np.random.uniform(0, 2 * np.pi, 12)
	result = minimize(huber_cost, x0, args=(rho, M1, M2, M3, M123))
	return result.fun


def parallel_minimize_huber(N, rho, M1, M2, M3, M123):
	# Use ProcessPoolExecutor for parallel execution
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(minimize_huber_cost, rho, M1, M2, M3, M123) for _ in range(N)]
		fun_list = [future.result() for future in futures]
	return fun_list


def huber_measure(rho, N=10):
	M1 = M(1)
	M2 = M(2)
	M3 = M(3)
	M123 = (M1 @ M2 @ M3)
	
	fun_list = parallel_minimize_huber(N, rho, M1, M2, M3, M123)
	
	return -2 * min(fun_list)


def helper_func(i, j, theta, phi, psi2, id):
	if i == 0:
		a = apply_kron([U(theta, phi), id, id])
	elif i == 1:
		a = apply_kron([id, U(theta, phi), id])
	else:
		a = apply_kron([id, id, U(theta, phi)])
	
	if j == 0:
		b = apply_kron([U(theta, phi), id, id])
	elif j == 1:
		b = apply_kron([id, U(theta, phi), id])
	else:
		b = apply_kron([id, id, U(theta, phi)])
	
	return np.kron(a, b) @ psi2


def chen_cost(params, rho, MM, M123, id):
	psi1 = [np.cos(params[0] / 2), np.exp(1j * params[1]) * np.sin(params[0] / 2)]
	psi2 = [np.cos(params[2] / 2), np.exp(1j * params[3]) * np.sin(params[2] / 2)]
	psi3 = [np.cos(params[4] / 2), np.exp(1j * params[5]) * np.sin(params[4] / 2)]
	psi = apply_kron([psi1, psi2, psi3])
	psi_2 = np.kron(psi, psi)
	rho2 = np.outer(rho, rho)
	
	ff = np.zeros((3, 3), dtype=object)
	for i in range(3):
		for j in range(i, 3):
			ff[i, j] = helper_func(i, j, params[6], params[7], psi_2, id)
	
	f = np.real(
		2 * np.sum([np.sqrt(ff[i, j].conj() @ rho2 @ M123 @ ff[i, j]) for i in range(3) for j in range(i + 1, 3)]) -
		2 * np.sum([np.sqrt(ff[i, j].conj() @ MM[i].T @ rho2 @ MM[i] @ ff[i, j]) for i in range(3) for j in
		            range(i + 1, 3)]) -
		np.sum([np.sqrt(ff[i, i].conj() @ MM[i].T @ rho2 @ MM[i] @ ff[i, i]) for i in range(3)])
	)
	
	return -f


def minimize_chen_cost(rho, MM, M123, id):
	x0 = np.random.uniform(0, 2 * np.pi, 8)
	result = minimize(chen_cost, x0, args=(rho, MM, M123, id))
	return result.fun


def parallel_minimize_chen(N, rho, MM, M123, id):
	# Use ProcessPoolExecutor for parallel execution
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(minimize_chen_cost, rho, MM, M123, id) for _ in range(N)]
		fun_list = [future.result() for future in futures]
	
	return fun_list


def chen_measure(rho, N=10):
	MM = [M(1), M(2), M(3)]
	M123 = MM[0] @ MM[1] @ MM[2]
	id = np.identity(2)
	
	fun_list = parallel_minimize_chen(N, rho, MM, M123, id)
	
	return -min(fun_list) / (2 * np.sqrt(2))


def gme_concurrence(state):
	rho = np.outer(state, np.conj(state))
	l = [partial_trace(rho, [0, 1]).data,
	     partial_trace(rho, [1, 2]).data,
	     partial_trace(rho, [0, 2]).data]
	return min(np.sqrt(2 * (1 - np.trace(rho @ rho))) for rho in l).real


def convex_roof_gme_concurrence(matrix):
	u, v = np.linalg.eigh(matrix)
	return np.sum(u_ * gme_concurrence(v_) for u_, v_ in zip(u, np.transpose(v)))


def gme(matrix):
	if not isinstance(matrix, np.ndarray):
		matrix = np.array(matrix)
	p00 = matrix[0, 0]
	p22 = matrix[2, 2]
	
	a = (1 - p00) ** 2 - p22 ** 2
	b = (1 - p00 - p22) * p22
	
	if a < 0:
		a = 0
	if b < 0:
		b = 0
	
	return min(np.sqrt(a), 2 * np.sqrt(b))


def concurrence(rho):
	yy_mat = np.fliplr(np.diag([-1, 1, 1, -1]))
	sigma = rho @ yy_mat @ rho.conj() @ yy_mat
	w = np.sqrt(np.maximum(np.sort(np.real(eigvals(sigma))), 0.0))
	return max(0.0, w[-1] - np.sum(w[0:-1]))
	

def concurrences(matrix):
	conc_12 = concurrence(partial_trace(matrix, [2]).data)
	conc_13 = concurrence(partial_trace(matrix, [1]).data)
	
	return conc_12, conc_13
