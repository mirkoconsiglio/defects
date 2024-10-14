"""

11.07.2023 - Riccarda Green project

@author: dzovan90@gmail.com

Code or any part of the code can not be used without permission for other projects.

ReducedDensityMatrix standalone code v5.

Only works with Periodic Boundary Conditions (PBC).

Computes the Reduced Density Matrix of smaller systems with Exact Diagonalization and Free Fermions and makes a
comparison.

This code is only useful for small systems of spin systems of size <20, for large one has comment out the exact
diagonalization parts and only use Free Fermion approach for N>20


"""
import json
import os
from collections import OrderedDict

import numpy as np
from matplotlib import rc

from utils import huber_measure, chen_measure

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)


# # this dumb function doesn't work when in dependencies.py file, so we put it here
# def doApplyHamClosed(psiIn):
# 	""" supplementary function  cast the Hamiltonian 'H' as a linear operator """
# 	return doApplyHamTENSOR_GENERAL(psiIn, hloc, L, True)


def doApplyHamTENSOR_GENERAL(psi: np.ndarray, hdisorder: np.ndarray, N: int, usePBC: bool, ):
	""" Basic succesive application of the local 4 leg local Hamiltonian to the psi tensor. This routine is designed
	to work with the LinearOperator function and the Lanczos type algorithms. Args: psi:        vector of length d**N
	describing the quantum state. hdisorder (ndarray):          array of ndim=4 describing the nearest neighbor
	coupling (local Hamiltonian of dimension (d^2, d^2)) however each pair is different depending on the disorder
	realization N:          the number of lattice sites. usePBC:     sets whether to include periodic boundary term.
	Returns: np.ndarray: state psi after application of the Hamiltonian.

		Logic of the application is explained somewhere in the notes. First, second and last application of the local
		Hamiltonian are written explicelty. The remaining steps are generic and the same algorithm is applied.
	"""
	# this is the local dimension
	d = 2
	
	# reshape the vector into a tensor with N legs each with dimension d
	reshape_array_default = np.full(N, d)
	psi = psi.reshape(reshape_array_default)
	
	# First application of the local Hamiltonian to the tensor quantum state for legs 1 and 2 (k = 0)
	psiOut = (hdisorder[0].reshape(4, 4) @ psi.reshape(4, 2 ** (N - 2))).reshape(reshape_array_default).reshape(2 ** N)
	
	# Second application of the local Hamiltonian to the tensor quantum state for legs 2 and 3 (k = 1)
	# generate the lists of transposition (first transpose different from the second one)
	transpose_array_default1 = np.arange(N)
	transpose_array_default2 = transpose_array_default1.copy()
	
	transpose_array_default1[0] = 1
	transpose_array_default1[1] = 2
	transpose_array_default1[2] = 0
	
	transpose_array_default2[0] = 2
	transpose_array_default2[1] = 0
	transpose_array_default2[2] = 1
	
	psiOut += (hdisorder[1].reshape(4, 4) @ psi.transpose(transpose_array_default1).reshape(4, 2 ** (N - 2))).reshape(
		reshape_array_default).transpose(transpose_array_default2).reshape(2 ** N)
	
	# Remaining applications of the local Hamiltonian  (k >= 2)
	for k in range(2, N - 1):
		transpose_array_default = np.arange(N)
		transpose_array_default[0] = k
		transpose_array_default[1] = k + 1
		transpose_array_default[k] = 0
		transpose_array_default[k + 1] = 1
		
		psiOut += (
				hdisorder[k].reshape(4, 4) @ psi.transpose(transpose_array_default).reshape(4, 2 ** (N - 2))).reshape(
			reshape_array_default).transpose(transpose_array_default).reshape(2 ** N)
	
	# PBC conditions application of the local hamiltonian on the end tails of the psi tensor
	if usePBC:
		# generate the lists of transposition (first transpose different from the second one)
		transpose_array_default1 = np.arange(N)
		transpose_array_default2 = transpose_array_default1.copy()
		
		transpose_array_default1[0] = N - 1
		transpose_array_default1[1] = 0
		transpose_array_default1[N - 1] = 1
		
		transpose_array_default2[0] = 1
		transpose_array_default2[1] = N - 1
		transpose_array_default2[N - 1] = 0
		
		psiOut += (hdisorder[N - 1].reshape(4, 4) @ psi.transpose(transpose_array_default1).reshape(4, 2 ** (
				N - 2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2 ** N)
	
	return psiOut


def PauliMatrice():
	"""
	Defining the Pauli matrices
	Arguments:

	Returns:
		sX (matrix, two-dimensional float):    sX matrix
		sY (matrix, two-dimensional float):    sY matrix
		sZ (matrix, two-dimensional float):    sZ matrix
		sI (matrix, two-dimensional float):    sI identity matrix

	"""
	
	sX = np.array([[0, 1.0], [1.0, 0]])
	sY = np.array([[0, -1.0j], [1.0j, 0]])
	sZ = np.array([[1.0, 0], [0, -1.0]])
	sI = np.array([[1.0, 0], [0, 1.0]])
	
	return sX, sY, sZ, sI


def IsingChainRiccarda(J: float, site_disorder: np.ndarray, htrans: float, hlong: float, N: int):
	"""
		Riccarda Green Project Hamiltonian - should correspond to the free-fermion tight-binding Hamiltonian

		THIS CAN BE USED ONLY FOR CHAIN WITH PERIODIC BOUNDARY CONDITIONS. THIS IS BECAUSE OF THE JORDAN-WIGNER
		CORRESPONDENCES BECAUSE IN CASE OF OPEN BOUNDARY CONDITIONS THE MAPPING IS NOT CONTROLLED WITH THE CHEMICAL
		POTENTIAL.

	"""
	
	sX, sY, sZ, sI = PauliMatrice()
	
	hloc = []
	for i in range(0, N):
		# elements
		prefactor = J / 2.0
		transverse = - 0.25 * htrans * (np.kron(sI, sZ) + np.kron(sZ, sI))
		longitudinal = - 0.25 * hlong * (np.kron(sI, sX) + np.kron(sX, sI))
		impurities = - 0.25 * (site_disorder[i] * np.kron(sZ, sI) + site_disorder[(i + 1) % N] * np.kron(sI, sZ))
		
		# total local hamiltonian expression
		htampon = (prefactor * (np.kron(sX, sX) + np.kron(sY, sY)) + transverse + longitudinal + impurities).reshape(2,
		                                                                                                             2,
		                                                                                                             2,
		                                                                                                             2)
		
		hloc.append(htampon)
	
	return hloc


# noinspection PyTypeChecker
def listaED_generate(state: np.ndarray, Nsites: int, l: int, m: int, n: int):
	"""

	Function that computes relevant spin correlation functins from the spin many-body wavefunction comming form Exact
	Diagonalization.

	Args: state(one-dimensional float array):                     many-body spin wave-function Nsites(integer): how
	many spins (we assume its d = 2, local dimension is 2) l (integer): subsystem site 1 m (integer): subsystem site 2
	n ( integer): subsystem site 3 Returns: lista(one-dimensional float array):                     list for the
	non-zero correlators from which one builds the reduced density matrix later

	"""
	
	# load in the Pauli matrices
	sX, sY, sZ, sI = PauliMatrice()
	
	listaED = np.zeros(20)
	
	# 1st element
	listaED[0] = 1.0
	
	# 2nd element
	listaED[1] = local_magnetization_general_point_general([sZ], [n], state, Nsites, 2).real
	
	# 3rd element
	listaED[2] = local_magnetization_general_point_general([sX, sX], [m, n], state, Nsites, 2).real
	
	# 4th element
	listaED[3] = local_magnetization_general_point_general([sY, sY], [m, n], state, Nsites, 2).real
	
	# 5th element
	listaED[4] = local_magnetization_general_point_general([sZ], [m], state, Nsites, 2).real
	
	# 6th element
	listaED[5] = local_magnetization_general_point_general([sZ, sZ], [m, n], state, Nsites, 2).real
	
	# 7th element
	listaED[6] = local_magnetization_general_point_general([sX, sX], [l, n], state, Nsites, 2).real
	
	# 8th element
	listaED[7] = local_magnetization_general_point_general([sX, sX], [l, m], state, Nsites, 2).real
	
	# 9th element
	listaED[8] = local_magnetization_general_point_general([sX, sX, sZ], [l, m, n], state, Nsites, 2).real
	
	# 10th element
	listaED[9] = local_magnetization_general_point_general([sX, sZ, sX], [l, m, n], state, Nsites, 2).real
	
	# 11th element
	listaED[10] = local_magnetization_general_point_general([sY, sY], [l, n], state, Nsites, 2).real
	
	# 12th element
	listaED[11] = local_magnetization_general_point_general([sY, sY], [l, m], state, Nsites, 2).real
	
	# 13th element
	listaED[12] = local_magnetization_general_point_general([sY, sY, sZ], [l, m, n], state, Nsites, 2).real
	
	# 14th element
	listaED[13] = local_magnetization_general_point_general([sY, sZ, sY], [l, m, n], state, Nsites, 2).real
	
	# 15th element
	listaED[14] = local_magnetization_general_point_general([sZ], [l], state, Nsites, 2).real
	
	# 16th element
	listaED[15] = local_magnetization_general_point_general([sZ, sZ], [l, n], state, Nsites, 2).real
	
	# 17th element
	listaED[16] = local_magnetization_general_point_general([sZ, sX, sX], [l, m, n], state, Nsites, 2).real
	
	# 18th element
	listaED[17] = local_magnetization_general_point_general([sZ, sY, sY], [l, m, n], state, Nsites, 2).real
	
	# 19th element
	listaED[18] = local_magnetization_general_point_general([sZ, sZ], [l, m], state, Nsites, 2).real
	
	# 20th element
	listaED[19] = local_magnetization_general_point_general([sZ, sZ, sZ], [l, m, n], state, Nsites, 2).real
	
	return listaED


def PartialTraceGeneralTensor(N: int, index_list: np.ndarray, A: np.ndarray):
	"""
	Function that computes the partial trace over index_list indices (the index list needs to be ordered from smaller
	to bigger index) Arguments: N (integer):                                        how many spins (local dimension d
	= 2) index_list (one-dimensional array of integers):     what is the subdomain indices we have to keep and the
	remaining ones get traced over A (one-dimensional float):                          state Returns: out (
	two-dimensional float):                        Reduced Density Matrix


	"""
	
	# reshape the input vectors into tensors (here we exploit the fact that psi* is just the complex conjugate of psi )
	reshape_array_default = np.full(N, 2)
	A_initial = A.reshape(reshape_array_default)
	
	# generate initial transpose indices vector (we apply permutations and operation so transposition is correctly
	# performed )
	list_A = np.arange(N)
	list_B = np.arange(N)
	
	# this changing the indices by one is because of python stuff (the numbering starts from zero and not 1)
	index_list = np.array(index_list) - 1
	
	##### generating the first transpose rule for A ###
	
	## initial step of moving the indices to the left
	for zz in range(0, len(index_list)):
		list_A[zz] = index_list[zz]
	
	## figure out what are the missing indices that happen because of overwriting in loop above
	list_A_no_dupl = list(OrderedDict.fromkeys(list_A))
	missing_indices = np.delete(np.arange(N), list_A_no_dupl)
	
	## now replace the doubled indices with indices in the missing_indices array
	counter = 0
	for zz in range(len(index_list), len(list_A)):
		for tt in range(0, len(index_list)):
			if list_A[zz] == index_list[tt]:
				list_A[zz] = missing_indices[counter]
				counter += 1
	
	##### generating the first transpose rule for B ###
	
	## initial step of moving the indices to the right
	for zz in range(0, len(index_list)):
		list_B[len(list_B) - zz - 1] = index_list[len(index_list) - zz - 1]
	
	## figure out what are the missing indices that happen because of overwriting in loop above
	list_B_no_dupl = list(OrderedDict.fromkeys(list_B))
	missing_indices = np.delete(np.arange(N), list_B_no_dupl)
	
	## now replace the doubled indices with indices in the missing_indices array
	counter = 0
	for zz in range(0, len(list_B) - len(index_list)):
		for tt in range(0, len(index_list)):
			if list_B[zz] == index_list[tt]:
				list_B[zz] = missing_indices[counter]
				counter += 1
	
	##### generating the second transpose rule for A ###
	
	list_A_cut = list_A[len(index_list):]
	list_A_cut_sort = np.sort(list_A_cut)
	
	list_B_cut = list_B[:-len(index_list)]
	list_B_cut_sort = np.sort(list_B_cut)
	
	transpose2_A = np.append(index_list, list_A_cut_sort)
	transpose2_B = np.append(list_B_cut_sort, index_list)
	
	############### MAIN OPERATION AFTER ALL PREPARATION HAS BEEN PERFORMED ::: TRANSPOSITION ON A and B
	A = A_initial.transpose(transpose2_A).reshape(2 ** len(index_list), 2 ** (N - len(index_list)))
	B = A_initial.transpose(transpose2_B).reshape(2 ** (N - len(index_list)), 2 ** len(index_list))
	
	# FINAL MULTIPLICATION
	out = (A @ np.conjugate(B))
	
	return out


def local_magnetization_general_point_general(sigma: np.ndarray, jlist: np.ndarray, psi: np.ndarray, N: int, d: int):
	"""

	Function to evaluate two point spin correlator < psi | sigma_{j1}^{x,y,z} sigma_{j2}^{x,y,z} ... sigma_{jN}^{x,y,
	z}| psi >

	ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi)

	Arguments:
		sigma (integer):                                    size of the fermionic chain
		jlist (one-dimensional integer array):              list of the position for the operators we want to evaluate
		psi (one-dimensional float array):                  state vector
		N (integer):                                        how many spins the wave-function is based on (assuming 2^d)
		d (integer):                                        local dimension
	Returns:
		mag (float):                                        value of the spin correlator

	"""
	
	initial_psi = psi.copy()
	
	# reshape into a tensor with N legs, each leg with dimension d
	reshape_array_default = np.full(N, d)
	psi = psi.reshape(reshape_array_default)
	
	# general loop
	for ii in range(0, len(jlist)):
		# transpose the tensor with N legs to match the position of the local operator
		transpose_array_default = np.arange(N)
		transpose_array_default[0] = jlist[ii] - 1
		transpose_array_default[jlist[ii] - 1] = 0
		
		# first operator application
		psi = (sigma[ii] @ psi.transpose(transpose_array_default).reshape(d, 2 ** (N - 1))).reshape(
			reshape_array_default).transpose(transpose_array_default)
	
	psi = psi.reshape(2 ** N)
	
	mag = np.conjugate(np.transpose(initial_psi)) @ psi
	
	return mag


# noinspection PyUnusedLocal
def generate_relevant_correlatorsAB(L: int, CM: np.ndarray, l: int, m: int, n: int):
	"""

	WE USE MAJORANA AB CORRELATION MATRICES INSTEAD OF FERMION CORRELATION MATRICES !!!

	Using the knowledge of all the relevant correlator we generate the irreducable list.

	There are 20 non-zero correlators.

	We first build the correlation matrix and then readout the values and things we need to build the spin operators.

	So this function finds the spin correlators and the corresponding fermion 2-point correlators for the reduced
	density matrix of three spins. The output of this function is a list for the non-zero correlators from which one
	builds the reduced density matrix later.

	First element of the list structure is the index, and the second one is the internal Mathematica file indecising
	and the final is the spin correlator at hand


	Args: L (integer):                                            systems size/matrix size CM (matrix array float)
	                            full majorana AB correlation matrix l (integer):
	                                 subsystem site 1 m (integer):
	                                 subsystem site 2 n (integer):
	                                 subsystem site 3 Returns: lista(one-dimensional float array):
	                                 list for the non-zero correlators from which one builds the reduced density
	                                 matrix later

	"""
	
	lista = np.zeros(20)
	
	alpha = m - l
	beta = n - m
	
	# 1st element: {{1, 1, {sI, sI, sI}}}
	lista[0] = 1.0
	
	# 2nd element {{2, 4, {sI, sI, sZ}}}
	lista[1] = CM[n - 1, n - 1]
	
	# 3rd element {{3, 6, {sI, sX, sX}}}
	lista[2] = ((-1.0) ** beta) * np.linalg.det(ABsubcorrelation_matrixXX(CM, m, n))
	
	# 4th element {{4, 11, {sI, sY, sY}}}
	lista[3] = ((-1.0) ** beta) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXX(CM, m, n)))
	
	# 5th element {{5, 13, {sI, sZ, sI}}}
	lista[4] = CM[m - 1, m - 1]
	
	# 6th element {{6, 16, {sI, sZ, sZ}}}
	lista[5] = np.linalg.det(ABsubcorrelation_matrixZZ(CM, m, n))
	
	# 7th element {{7, 18, {sX, sI, sX}}}
	lista[6] = ((-1.0) ** (alpha + beta)) * np.linalg.det(ABsubcorrelation_matrixXX(CM, l, n))
	
	# 8th element {{8, 21, {sX, sX, sI}}}
	lista[7] = ((-1.0) ** alpha) * np.linalg.det(ABsubcorrelation_matrixXX(CM, l, m))
	
	# 9th element {{9, 24, {sX, sX, sZ}}}
	lista[8] = ((-1.0) ** (alpha + 1.0)) * np.linalg.det(ABsubcorrelation_matrixXXZ(CM, l, m, n))
	
	# 10th element {{10, 30, {sX, sZ, sX}}}
	lista[9] = ((-1.0) ** (alpha + beta)) * np.linalg.det(ABsubcorrelation_matrixXZX(CM, l, m, n))
	
	# 11th element {{11, 35, {sY, sI, sY}}}
	lista[10] = ((-1.0) ** (alpha + beta)) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXX(CM, l, n)))
	
	# 12th element {{12, 41, {sY, sY, sI}}}
	lista[11] = ((-1.0) ** alpha) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXX(CM, l, m)))
	
	# 13th element {{13, 44, {sY, sY, sZ}}}
	lista[12] = ((-1.0) ** (alpha + 1.0)) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXXZ(CM, l, m, n)))
	
	# 14th element {{14, 47, {sY, sZ, sY}}}
	lista[13] = ((-1.0) ** (alpha + beta)) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXZX(CM, l, m, n)))
	
	# 15th element {{15, 49, {sZ, sI, sI}}}
	lista[14] = CM[l - 1, l - 1]
	
	# 16th element {{16, 52, {sZ, sI, sZ}}}
	lista[15] = np.linalg.det(ABsubcorrelation_matrixZZ(CM, l, n))
	
	# 17th element {{17, 54, {sZ, sX, sX}}}
	lista[16] = ((-1.0) ** (beta + 1.0)) * np.linalg.det(ABsubcorrelation_matrixZXX(CM, l, m, n))
	
	# 18th element {{18, 59, {sZ, sY, sY}}}
	lista[17] = ((-1.0) ** (beta + 1.0)) * np.linalg.det(np.transpose(ABsubcorrelation_matrixZXX(CM, l, m, n)))
	
	# 19th element {{19, 61, {sZ, sZ, sI}}}
	lista[18] = np.linalg.det(ABsubcorrelation_matrixZZ(CM, l, m))
	
	# 20th element {{20, 64, {sZ, sZ, sZ}}}
	lista[19] = -np.linalg.det(ABsubcorrelation_matrixZZZ(CM, l, m, n))
	
	return lista


def RDM_FF(lista: np.ndarray) -> np.ndarray:
	"""

	Using the knowledge of how the 3 spins sized subsystem reduced density matrix looks we build it term by term.
	The exact for of the matrix is given in the Mathematica file.

	Args: lista(one-dimensional float array):                     list for the non-zero correlators from which one
	builds the reduced density matrix later

	Returns:
		RDM(matrix, two-dimensional float array):             Reduced Density Matrix for Free Fermions

	"""
	
	# generate an empty matrix
	RDM = np.zeros([8, 8])
	
	#  manually fill out the non-zero elements
	# 1st row
	RDM[0, 0] = lista[0] + lista[4] + lista[5] + lista[1] + lista[14] + lista[15] + lista[18] + lista[19]
	RDM[0, 3] = -lista[3] + lista[16] - lista[17] + lista[2]
	RDM[0, 5] = lista[6] + lista[9] - lista[10] - lista[13]
	RDM[0, 6] = lista[7] + lista[8] - lista[11] - lista[12]
	
	# 2nd row
	RDM[1, 1] = lista[0] + lista[4] - lista[5] - lista[1] + lista[14] - lista[15] + lista[18] - lista[19]
	RDM[1, 2] = lista[3] + lista[16] + lista[17] + lista[2]
	RDM[1, 4] = lista[6] + lista[9] + lista[10] + lista[13]
	RDM[1, 7] = lista[7] - lista[8] - lista[11] + lista[12]
	
	# 3rd row
	RDM[2, 1] = lista[3] + lista[16] + lista[17] + lista[2]
	RDM[2, 2] = lista[0] - lista[4] - lista[5] + lista[1] + lista[14] + lista[15] - lista[18] - lista[19]
	RDM[2, 4] = lista[7] + lista[8] + lista[11] + lista[12]
	RDM[2, 7] = lista[6] - lista[9] - lista[10] + lista[13]
	
	# 4th row
	RDM[3, 0] = -lista[3] + lista[16] - lista[17] + lista[2]
	RDM[3, 3] = lista[0] - lista[4] + lista[5] - lista[1] + lista[14] - lista[15] - lista[18] + lista[19]
	RDM[3, 5] = lista[7] - lista[8] + lista[11] - lista[12]
	RDM[3, 6] = lista[6] - lista[9] + lista[10] - lista[13]
	
	# 5th row
	RDM[4, 1] = lista[6] + lista[9] + lista[10] + lista[13]
	RDM[4, 2] = lista[7] + lista[8] + lista[11] + lista[12]
	RDM[4, 4] = lista[0] + lista[4] + lista[5] + lista[1] - lista[14] - lista[15] - lista[18] - lista[19]
	RDM[4, 7] = -lista[3] - lista[16] + lista[17] + lista[2]
	
	# 6th row
	RDM[5, 0] = lista[6] + lista[9] - lista[10] - lista[13]
	RDM[5, 3] = lista[7] - lista[8] + lista[11] - lista[12]
	RDM[5, 5] = lista[0] + lista[4] - lista[5] - lista[1] - lista[14] + lista[15] - lista[18] + lista[19]
	RDM[5, 6] = lista[3] - lista[16] - lista[17] + lista[2]
	
	# 7th row
	RDM[6, 0] = lista[7] + lista[8] - lista[11] - lista[12]
	RDM[6, 3] = lista[6] - lista[9] + lista[10] - lista[13]
	RDM[6, 5] = lista[3] - lista[16] - lista[17] + lista[2]
	RDM[6, 6] = lista[0] - lista[4] - lista[5] + lista[1] - lista[14] - lista[15] + lista[18] + lista[19]
	
	# 8th row
	RDM[7, 1] = lista[7] - lista[8] - lista[11] + lista[12]
	RDM[7, 2] = lista[6] - lista[9] - lista[10] + lista[13]
	RDM[7, 4] = -lista[3] - lista[16] + lista[17] + lista[2]
	RDM[7, 7] = lista[0] - lista[4] + lista[5] - lista[1] - lista[14] + lista[15] + lista[18] - lista[19]
	
	# normalization of the values comes due to the definitions (so we normalize in this case with 2^{-3})
	RDM = np.array(RDM) / 8.0
	
	return RDM


# noinspection PyUnusedLocal
def generate_relevant_correlatorsAB_2(L: int, CM: np.ndarray, l: int, m: int):
	"""

	WE USE MAJORANA AB CORRELATION MATRICES INSTEAD OF FERMION CORRELATION MATRICES !!!

	THIS ONLY FOR THE TWO SPIN REDUCED DENSITY MATRIX.

	Using the knowledge of all the relevant correlator we generate the irreducable list.

	There are 20 non-zero correlators.

	We first build the correlation matrix and then readout the values and things we need to build the spin operators.

	So this function finds the spin correlators and the corresponding fermion 2-point correlators for the reduced
	density matrix of three spins. The output of this function is a list for the non-zero correlators from which one
	builds the reduced density matrix later.

	First element of the list structure is the index, and the second one is the internal Mathematica file indecising
	and the final is the spin correlator at hand


	Args: L (integer):                                            systems size/matrix size CM (matrix array float)
	                            full majorana AB correlation matrix l (integer):
	                                 subsystem site 1 m (integer):
	                                 subsystem site 2 Returns: lista(one-dimensional float array):
	                                 list for the non-zero correlators from which one builds the reduced density
	                                 matrix later

	"""
	
	lista = np.zeros(6)
	
	alpha = m - l
	
	# 1st element: {{1, 1, {sI, sI}}} ok
	lista[0] = 1.0
	
	# 2nd element {{2, 4, {sI, sZ}}} ok
	lista[1] = CM[m - 1, m - 1]
	
	# 3rd element {{3, 6, {sX, sX}}} ok
	lista[2] = ((-1.0) ** alpha) * np.linalg.det(ABsubcorrelation_matrixXX(CM, l, m))
	
	# 4th element {{4, 11, {sY, sY}}} ok
	lista[3] = ((-1.0) ** alpha) * np.linalg.det(np.transpose(ABsubcorrelation_matrixXX(CM, l, m)))
	
	# 5th element {{5, 13, {sZ, sI}}} ok
	lista[4] = CM[l - 1, l - 1]
	
	# 6th element {{6, 16, {sZ, sZ}}} ok
	lista[5] = np.linalg.det(ABsubcorrelation_matrixZZ(CM, l, m))
	
	return lista


def RDM_FF_2(lista: np.ndarray):
	"""

	Using the knowledge of how the 2 spins sized subsystem reduced density matrix looks we build it term by term.
	The exact for of the matrix is given in the Mathematica file.

	Args: lista(one-dimensional float array):                   list for the non-zero correlators from which one
	builds the reduced density matrix later

	Returns:
		RDM(matrix, two-dimensional float array):             Reduced Density Matrix for Free Fermions

	"""
	
	# generate an empty matrix
	RDM = np.zeros([4, 4])
	
	#  manually fill out the non-zero elements
	# 1st row
	RDM[0, 0] = lista[0] + lista[4] + lista[5] + lista[1]
	RDM[0, 3] = -lista[3] + lista[2]
	
	RDM[1, 1] = lista[0] + lista[4] - lista[5] - lista[1]
	RDM[1, 2] = lista[3] + lista[2]
	
	RDM[2, 1] = lista[3] + lista[2]
	RDM[2, 2] = lista[0] - lista[4] - lista[5] + lista[1]
	
	RDM[3, 0] = -lista[3] + lista[2]
	RDM[3, 3] = lista[0] - lista[4] + lista[5] - lista[1]
	
	# normalization of the values comes due to the definitions (so we normalize in this case with 2^{-2})
	RDM = np.array(RDM) / 4.0
	
	return RDM


def ABsubcorrelation_matrixXX(CM: np.ndarray, l: int, m: int):
	# generate the list of indices for the matrix
	
	# corresponds to the annihilation operator index
	listA = list(range(l + 1, m + 1))
	
	# corresponds to the creation operator index
	listB = list(range(l, m))
	
	# generate the matrix
	matrix = np.zeros([len(listA), len(listA)])
	
	for i in range(0, len(listA)):
		for j in range(0, len(listB)):
			matrix[i, j] = -CM[listA[i] - 1, listB[j] - 1]
	
	return matrix


def ABsubcorrelation_matrixZZ(CM: np.ndarray, index1: int, index2: int):
	matrix = np.zeros([2, 2])
	
	matrix[0, 0] = -CM[index1 - 1, index1 - 1]
	matrix[0, 1] = -CM[index1 - 1, index2 - 1]
	matrix[1, 0] = -CM[index2 - 1, index1 - 1]
	matrix[1, 1] = -CM[index2 - 1, index2 - 1]
	
	return matrix


def ABsubcorrelation_matrixZZZ(CM: np.ndarray, index1: int, index2: int, index3: int):
	matrix = np.zeros([3, 3])
	matrix[0, 0] = -CM[index1 - 1, index1 - 1]
	matrix[0, 1] = -CM[index1 - 1, index2 - 1]
	matrix[0, 2] = -CM[index1 - 1, index3 - 1]
	
	matrix[1, 0] = -CM[index2 - 1, index1 - 1]
	matrix[1, 1] = -CM[index2 - 1, index2 - 1]
	matrix[1, 2] = -CM[index2 - 1, index3 - 1]
	
	matrix[2, 0] = -CM[index3 - 1, index1 - 1]
	matrix[2, 1] = -CM[index3 - 1, index2 - 1]
	matrix[2, 2] = -CM[index3 - 1, index3 - 1]
	
	return matrix


def ABsubcorrelation_matrixXXZ(CM: np.ndarray, l: int, m: int, n: int):
	# generate the list of indices for the matrix
	
	# corresponds to the annihilation operator index
	listA = list(range(l, m + 1))
	listA.remove(m)
	listA.append(n)
	
	# corresponds to the creation operator index
	listB = list(range(l + 1, m + 1))
	listB.append(n)
	
	# generate the matrix
	matrix = np.zeros([len(listA), len(listA)])
	
	for i in range(0, len(listA)):
		for j in range(0, len(listA)):
			matrix[i, j] = -CM[listA[i] - 1, listB[j] - 1]
	
	return matrix


def ABsubcorrelation_matrixXZX(CM: np.ndarray, l: int, m: int, n: int):
	# generate the list of indices for the matrix
	
	# corresponds to the annihilation operator index
	listA = list(range(l, n + 1))
	listA.remove(m)
	listA.remove(n)
	
	# corresponds to the creation operator index
	listB = list(range(l, n + 1))
	listB.remove(m)
	listB.remove(l)
	
	# generate the matrix
	matrix = np.zeros([len(listA), len(listA)])
	
	for i in range(0, len(listA)):
		for j in range(0, len(listA)):
			matrix[i, j] = -CM[listA[i] - 1, listB[j] - 1]
	
	return matrix


def ABsubcorrelation_matrixZXX(CM: np.ndarray, l: int, m: int, n: int):
	# generate the list of indices for the matrix
	
	# corresponds to the annihilation operator index
	listA = [l]
	listA.extend(list(range(m + 1, n + 1)))
	
	# corresponding to the creating index
	listB = [l]
	listB.extend(list(range(m, n)))
	
	matrix = np.zeros([len(listA), len(listB)])
	
	for i in range(0, len(listA)):
		for j in range(0, len(listB)):
			matrix[i, j] = -CM[listA[i] - 1, listB[j] - 1]
	
	return matrix


def A_matrix_choice1(L: int, J: float, h: float, disorder_array: np.ndarray):
	"""

	LSM A matrix definition and choice depending on the lower energy ground state

	"""
	matrix = np.zeros([L, L])
	
	# tridiagonal structure
	for i in range(L):
		for j in range(L):
			
			# diagonal
			if i == j:
				matrix[i, j] = -(h + disorder_array[i])
			
			# first lower offdiagonal
			if i == j + 1:
				matrix[i, j] = J
			
			# first upper offdiagonal
			if i == j - 1:
				matrix[i, j] = J
	
	matrix[0, L - 1] = J
	matrix[L - 1, 0] = J
	
	return matrix


def A_matrix_choice2(L: int, J: float, h: float, disorder_array: np.ndarray):
	"""

	LSM A matrix definition and choice depending on the lower energy ground state

	"""
	matrix = np.zeros([L, L])
	
	# tridiagonal structure
	for i in range(L):
		for j in range(L):
			
			# diagonal
			if i == j:
				matrix[i, j] = -(h + disorder_array[i])
			
			# first lower offdiagonal
			if i == j + 1:
				matrix[i, j] = J
			
			# first upper offdiagonal
			if i == j - 1:
				matrix[i, j] = J
	
	matrix[0, L - 1] = -J
	matrix[L - 1, 0] = -J
	
	return matrix


############################## MAIN CODE ###########################
def main():
	# system parameters
	L = 1024
	J = 1.0
	h = 2.0
	mid_defect = L // 2
	
	# position and strength of impurities definitions
	eps_max = 5
	steps = eps_max * 10 + 1
	
	os.makedirs(f'h_{h:.1f}', exist_ok=True)
	
	for d in range(1, 10):
		print(f'd = {d}')
		end = eps_max / d
		eps_list = np.linspace(0, end, steps).tolist()
		l = mid_defect - d
		m = mid_defect
		n = mid_defect + d
		data = []
		for eps in eps_list:
			epsilon_l = -eps
			epsilon_m = -eps
			epsilon_n = -eps
			
			disorder_array = np.zeros(L)
			disorder_array[l - 1] = epsilon_l
			disorder_array[m - 1] = epsilon_m
			disorder_array[n - 1] = epsilon_n
			print(f'eps_d = {eps * d}')
			
			############ FREE FERMIONIC CODE USING LIEB-SCHULTZ-MATTIS APPROACH ##########
			
			# GENERATE TWO A MATRICES AND DECIDE ON THE PROPER GROUND-STATE
			A1 = A_matrix_choice1(L, J, h, disorder_array)
			lambda_square1, phik1 = np.linalg.eigh(A1 @ A1)
			energy1 = -0.5 * np.sum(np.sqrt(lambda_square1))
			
			A2 = A_matrix_choice2(L, J, h, disorder_array)
			lambda_square2, phik2 = np.linalg.eigh(A2 @ A2)
			energy2 = -0.5 * np.sum(np.sqrt(lambda_square2))
			
			# print(energy1, energy2, energy1 - energy2)
			
			if energy1 < energy2 or np.isnan(energy2):
				lambda_square = lambda_square1.copy()
				phik = phik1.copy()
				A = A1.copy()
			else:
				lambda_square = lambda_square2.copy()
				phik = phik2.copy()
				A = A2.copy()
			
			psik = np.zeros((L, L))
			for i in range(L):
				psik[:, i] = (1.0 / np.sqrt(lambda_square[i])) * (phik[:, i] @ A)
			
			# generate G matrix
			# here python inverts row and columns, so we do an additional transposition compared to the notes
			G_correlation_matrix = np.transpose(-psik @ np.transpose(phik))
			
			# generate the list of relevant fermionic correlators
			listaFF = generate_relevant_correlatorsAB(L, G_correlation_matrix, l, m, n)
			
			# generate the Reduced Density Matrix
			RDM_fermionic = RDM_FF(listaFF)
			
			# print it out and round of the values for better visual presentation in the terminal
			# print('-------------------- FREE FERMIONS RDM ----------------')
			matrix = RDM_fermionic.real
			
			# u, v = np.linalg.eigh(matrix)
			#
			# print(np.matrix.round(matrix, 3))
			# print(u)
			# print(np.matrix.round(v, 3))
			# quit()
			
			data.append(dict(d=d, eps_d=eps * d, matrix=matrix.tolist()))
			
		with open(f'h_{h:.1f}/d_{d}.json', 'w') as json_file:
			json.dump(data, json_file, indent=4)


if __name__ == '__main__':
	# main()
	
	directory = 'h_2.0'
	
	for file in os.listdir(directory):
		if file.endswith('.json'):
			file_path = os.path.join(directory, file)

			with open(file_path, 'r') as json_file:
				data = json.load(json_file)

			for i, _data in enumerate(data):
				print(_data['d'], _data['eps_d'])

				matrix = _data['matrix']

				data[i]['chen_measure'] = chen_measure(matrix)

			with open(file_path, 'w') as json_file:
				json.dump(data, json_file, indent=4)
