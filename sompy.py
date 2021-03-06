from __future__ import division

'''
Self Organizing Map using numpy
This code is licensed and released under the GNU GPL

Author: Roland Halbig
Email:	halbig.roland@freenet.de
Date:	Jan 10 2014

This implementation uses a 2D square grid but can easily be adjusted to other grids.
Just inherit the class an reimplement all node access functions and the initialization procedure.

This code is a fork from Kyle Dickerson's sompy.py (kyle.dickerson@gmail.com). 
His branch of sompy can be found on github: 
https://github.com/kdickerson/Sompy

If you do use this code for something, please send me a short message. 
I'd be happy if anybody could use it for his or her endevour!
'''

import random
import sys
from math import *
import numpy as np
from PIL import Image
import scipy.ndimage

########################################################################################################
## distance measures
def eucl_distance(a, b):
	axis = len(np.shape(a))-1
	return np.sqrt(((a - b)**2).sum(axis=axis))
def L1_distance(r1, r2):
	axis = len(np.shape(r1))-1
	return (abs(r1 - r2)).sum(axis=axis)
## 
def get_bounding_box(samples):
	samples = np.asarray(samples)
	minmax = np.asarray( [ [0.0]*2 ] * np.shape(samples)[1])
	for i in range(np.shape(samples)[1]):
		minmax[i,0] = min(samples[:,i])
		minmax[i,1] = max(samples[:,i])
	return minmax

########################################################################################################
## SOM class
class SOM():
	def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005, initial_radius=None, init_mode='random'):
		''' 
		Self Organizing Map. This implementation uses a rectengular grid. 
		Additional functionalities, such as a counting array for the choices as Best Matching Unit and for the removal of Nodes are implemented. 
		
		init_mode - values:
		'random' : random vectors with values within the bounding box of the data
		'xy_box': regular grid on the x-y plane. This functionality should be used on PCA-transformed data! 
		'1d_line': regular line on the x-y plane. 
		'''
		self.height, self.width, self.FV_size, self.learning_rate, self.init_mode = height, width, FV_size, learning_rate, init_mode
		self.total = self.width * self.height
		self.radius = (height+width)/3 if initial_radius is None else initial_radius

		self.nodes = [[]] 				# saves node data
		self.indexMap = [[]] 			# saves index inside kohonen map
		self.BMUcount = [0]*self.total 	# counts bmu selection for each node
		self.nodeIndex = [] 			# saves index of nodes. Useful for adding or removing nodes
		
		self.residual = []
	
	def train(self, train_vector=[[]], iterations=100, num_samples = 1, cont=False, res=False, disp=True):
		'''
		Training of the SOM. Parameters:
		
		train_vector	- the training data
		iterations 	- number of iterations
		num_samples 	- determines the number of samples to be approximated in each iteration
		cont 		- continue training without reinitializing the SOM 
		res  		- calculate residual and return a list of all residua
		disp 		- display training progress 
		'''
		train_vector = np.asarray(train_vector)
		max_iterations = int(iterations)
		res_list = []
		current_radius = self.radius
		current_rate = self.learning_rate
		if not cont: 
			self._initialize(train_vector, self.init_mode)	
		
		for iteration in range(1, max_iterations+1):
			if disp:
				sys.stdout.write("\rTraining Iteration: %d/%d" %(iteration, max_iterations)); 
				sys.stdout.flush()
			
			current_radius = self.radius * self._decay_function(iteration, max_iterations)
			current_rate = self.learning_rate * self._decay_function(iteration, max_iterations)
			
			if res: 
				r = self._get_residual(train_vector, iteration, max_iterations)
				self.residual.append(r)
				if iteration > max_iterations/4 and len(self.residual) > 1 and np.abs(self.residual[-1]-self.residual[-2]) < 0.1:
					break;
			
			for j in range(num_samples):
				j = int( np.random.uniform(0, len(train_vector),1))
				best = self._get_best_match(train_vector[j])				
				for [neighbor, dist] in self._get_neighborhood(best, current_radius):
					influence = self._get_kernel(dist, current_radius, iteration)
					self.nodes[neighbor,:] += current_rate * influence * (train_vector[j][:] - self.nodes[neighbor,:])

		if disp: sys.stdout.write("\n")
		return res_list;

	def get_grid(self):
		''' 
		Returns the grid containing the prototype vectors 
		'''
		return self._get_data_as_grid(self.nodes)
		
	def get_data(self):
		'''
		Returns the following three data structures:
		nodes	- list of the prototypes
		idcs		- list of the corresponding SOM index
		bmus		- list of the corresponding BMU counts
		'''
		idcs, nodes, bmus = [], [], []
		for i in self.nodeIndex:
			idcs.append(self.indexMap[i])
			nodes.append(self.nodes[self.nodeIndex[i]])
			bmus.append(self.BMUcount[self.nodeIndex[i]])
		idcs = np.asarray(idcs)
		nodes = np.asarray(nodes)
		bmus = np.asarray(bmus)
		return nodes, idcs, bmus
		
	def get_residual(self):
		return self.residual
		
	def remove_bmu_nodes(self, threshold): 
		''' 
		You may want to do this before getting the data. 
		This removes entries from the list of nodes with BMU count smaller than threshold. 
		'''
		for n in dict(self.nodeIndex):
			index = self.nodeIndex[n]
			if self.BMUcount[index] < threshold:
				self.nodeIndex.pop(index)
				
	def save_similarity_mask(self, filename, threshold=0, path="./"):
		'''
		Write the similarity mask to a file.
		threshold - does thresholding for values from 0 to 255
		'''
		tmp_nodes = self._get_distance_mask()
		self._save_mask(tmp_nodes, filename, path, threshold=threshold)
	
	def save_bmu_mask(self, filename, path="./"):
		'''
		Write the BMU mask to a file.
		The ligther the value the more often a node has been chosen as Best Matching Unit.
		'''
		tmp_nodes = np.asarray(self.BMUcount)
		self._save_mask(tmp_nodes, filename, path)
	
	###################################
	###### private functions
	
	## parameter functions
	def _get_kernel(self, distance, radius, iteration):
		return exp( -1.0 * abs(distance) / (2*radius*iteration) )		
	
	def _decay_function(self, iteration, max_iterations):
		return 1-np.tanh((iteration)/max_iterations)
	
	## node access method for a rectangular grid
	def _get_index(self, i, j):
		return j + i*self.width

	def _get_best_match(self, target_FV): 
		'''Returns location of best match, uses Euclidean distance '''
		best = np.argmin(L1_distance(self.nodes, target_FV))
		self._raise_bmu_count(best)
		return best
		
	def _get_neighborhood(self, index, radius): 
		''' Returns a list of points which live within chessboard distance 'dist' of 'pt'. pt is (row, column). '''
		if radius < 1: 
			return [[index, 0.0]]
		else:
			radius = int(radius)
		pt = self.indexMap[index]
		min_y = max(int(pt[0] - radius), 0)
		max_y = min(int(pt[0] + radius + 1), self.height)
		min_x = max(int(pt[1] - radius), 0)
		max_x = min(int(pt[1] + radius + 1), self.width)
		neighbors = []
		for y in range(min_y, max_y):
			for x in range(min_x, max_x):
				''' manhattan distance '''
				#dist = (abs(y-pt[0])**2 + abs(x-pt[1])**2)**0.5
				'''euclidean distance'''
				dist = np.sqrt(abs(y-pt[0])**2 + abs(x-pt[1])**2)
				neighbors.append([ self._get_index(y,x), dist ])
		return neighbors
		
	def _initialize(self, train_vector, mode='random'): 
		''' initializes nodes as numpy arrays using init_mode '''
		self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
		self.indexMap = [[0,0]]*self.total
		self.nodeIndex = dict()
		for i in range(self.height):
			for j in range(self.width):
				index = self._get_index(i,j)
				self.indexMap[index] = [i, j]	
				self.nodeIndex[index] = index		
		if mode == 'random':
			''' random vectors with values within the bounding box of the data '''
			minmax = get_bounding_box(train_vector)
			self.nodes = np.asarray( map( lambda x: np.random.uniform(np.min(minmax),np.max(minmax), (self.FV_size,)) , self.nodes ) )
		elif mode == 'xy_box':
			''' regular grid on the x-y plane. This functionality should be used on PCA-transformed data! '''
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = get_bounding_box(train_vector)
			hh = 1.0/float(self.height-1) if (self.height > 1) else 1
			hw = 1.0/float(self.width-1) if (self.width > 1) else 1
			for i in range(self.height):
				for j in range(self.width):
					index = self._get_index(i,j)
					self.nodes[index, 0] = minmax[0,0] + i*hh*abs(minmax[0,1]-minmax[0,0])
					self.nodes[index, 1] = minmax[1,0] + j*hw*abs(minmax[1,1]-minmax[1,0])			
		elif mode == '1d_line':
			''' regular line on the x-y plane. '''
			if self.width > 1 and self.height > 1:
				print "Warning: In _initialize: cannot initialize 2D array as 1D line! Using random values!\n"
				self._initialize(train_vector, 'random')
				return
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = get_bounding_box(train_vector)
			dimension = self.total#max(self.height, self.width)
			h = 1.0/float(dimension-1) if (dimension > 1) else 1
			for i in range(dimension):
					index = self._get_index(i,0) if self.width < self.height else self._get_index(0,i)
					self.nodes[index,:] = minmax[:,0] + i*h*np.abs(minmax[:,0]-minmax[:,1])
		elif len(mode) == 1:
			''' random vectors with values within the range of [[min, max]] '''
			range0, range1 = mode[0][0], mode[0][1]
			self.nodes = np.asarray( map( lambda x: np.random.uniform(range0,range1,(self.FV_size,)) , self.nodes ) )
		else:
			raise NotImplementedError( "ERROR! in SOM: Initialization mode not known!" )
		return;
	
	def _get_data_as_grid(self, data): 
		'''Return 2D representation of the data. '''
		N = np.zeros((self.height, self.width), float)
		for index in range(self.total):
			(i, j) = self.indexMap[index]
			N[i,j] = data[index]
		return N
	
	def _get_residual(self, samples, iteration, max_iterations):
		res = 0.0
		
		radius = self.radius * self._decay_function(iteration, max_iterations)
		rate = self.learning_rate * self._decay_function(iteration, max_iterations)
		for j in range(len(samples)):
			best = self._get_best_match(samples[j])			
			for [neighbor, dist] in self._get_neighborhood(best, radius):
				influence = self._get_kernel(dist, radius, iteration)
				res += rate * influence * eucl_distance(samples[j][:], self.nodes[neighbor,:])
#			res += eucl_distance(samples[j][:], self.nodes[best,:])
		return res
	
		
	####################################################
	## methods for counting the Best Matching Unit Index
	def _reset_bmu_count(self):
		map( lambda x : 0, self.BMUcount )
		
	def _raise_bmu_count(self, index):
		self.BMUcount[index] += 1

	def _get_bmu_grid(self):
		return self._get_data_as_grid(self.BMUcount)
		
	####################################################
	## The SOM's similarity mask 
	## which is some kind of average gradient. 
	## The darker the area the more rapid the change.
		
	def _get_distance_mask(self):
		tmp_nodes = np.zeros((self.total,), float)
		for r in range(self.height):
			for c in range(self.width):
				index = self._get_index(r,c)
				N = self.nodes[index,:]
				neighborhood = self._get_neighborhood(index, 1)
				for [n, dist] in neighborhood:
					tmp_nodes[n] += eucl_distance(self.nodes[n,:], N)
				tmp_nodes[self._get_index(r,c)] = tmp_nodes[self._get_index(r,c)]**0.01
		return tmp_nodes
		
# save grid values to greyscale image
	def _get_similarity_mask(self):
		tmp_nodes = self._get_distance_mask()
		return self._get_data_as_grid(tmp_nodes)
		
	def _save_mask(self, tmp_nodes, filename, path="./", threshold=0):
		tmp_nodes -= tmp_nodes.min()
		tmp_nodes *= 255 / max(1, tmp_nodes.max())
		tmp_nodes = 255 - tmp_nodes
		if threshold > 0:
			for i in range(len(tmp_nodes)):
				tmp_nodes[i] = 0 if tmp_nodes[i] < threshold else 255
		img = Image.new("L", (self.height,self.width))
		for r in range(self.height):
			for c in range(self.width):
				#print r, c, self.width, self.height
				img.putpixel((r,c), 0)
		for i in self.nodeIndex:
			value = tmp_nodes[i]
			img.putpixel(self.indexMap[i], value)
		scale = 20
#		img = img.resize((self.width*int(scale),self.height*int(scale)),Image.NEAREST)
		img.save(path + filename + ".pgm")
	
################################################################################################

class TorusSOM(SOM):
	''' 
	This SOM uses a rectengular grid projected on the surface of a torus.
	The code was originally written by Kyle Dickerson.
	'''
	def _get_neighborhood(self, index, radius):
		if radius < 1: 
			return [[index, 0.0]]
		else:
			radius = int(radius)
		pt = self.indexMap[index]
		# This allows the grid to wrap vertically and horizontally
		min_y = int(pt[0] - radius)
		max_y = int(pt[0] + radius)+1
		min_x = int(pt[1] - radius)
		max_x = int(pt[1] + radius)+1
		
		if self.width == 1:
			min_x, max_x = 0, 1
		if self.height== 1:
			min_y, max_y = 0, 1
	
		# just build the cross product of the bounds
		neighbors = []
		for y in range(min_y, max_y):
			y_piece = (y-pt[0])**2
			oldy = y
			y = y + self.height if y < 0 else y % self.height
			for x in range(min_x, max_x):
				# Manhattan
				# d = abs(y-pt[0]) + abs(x-pt[1])
				# Euclidean
				d = (y_piece + (x-pt[1])**2)**0.5
				x = x + self.width if x < 0 else x % self.width
				neighbors.append([ self._get_index(y,x), d ])
		return neighbors

		
class LinearSOM(TorusSOM):
	''' 
	This SOM uses linearily decreasing decay functions.
	The code was originally written by Kyle Dickerson.
	'''
	def _decay_function(self, iteration, max_iterations):
		return 1 + ((max_iterations - iteration) / max_iterations)



