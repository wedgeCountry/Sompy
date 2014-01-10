from __future__ import division

## Roland Halbig
## halbig.roland@freenet.de
## Dec 19 2013

## Self-organizing map using numpy
## This code is licensed and released under the GNU GPL

## This code uses a square grid but can be adjusted to a hexagonal grid. 
## In order to do this, change the initialization an the find_neighborhood functions.
##
## This code is a fork from sompy.py by Kyle Dickerson (kyle.dickerson@gmail.com). 
## His branch of sompy can be found on github:
## https://github.com/kdickerson/Sompy

## If you do use this code for something, please let me know, I'd like to know if has been useful to anyone.

import random
from math import *
import sys
from PIL import Image
import scipy.ndimage
import numpy as np

import sklearn.decomposition

########################################################################################################

def getBoundingBox(samples):
	samples = np.asarray(samples)
	minmax = np.asarray( [ [0.0]*2 ] * np.shape(samples)[1])
	for i in range(np.shape(samples)[1]):
		minmax[i,0] = min(samples[:,i])
		minmax[i,1] = max(samples[:,i])
	return minmax

class SOM():
	''' Self Organizing Map Class. This implementation uses a rectengular grid. Other functionalities, such as
	a counting array for the choices as Best Matching Unit and for the removal of Nodes are implemented. '''
	def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005, initial_radius=None, FV_ranges='random'):
		self.height = height
		self.width = width
		self.total = self.width * self.height
		self.FV_size = FV_size
		self.FV_ranges = FV_ranges
		self.learning_rate = learning_rate
		self.radius = (height+width)/3 if initial_radius is None else initial_radius

		self.nodes = [] 				# saves node data
		self.nodeDict = [] 				# saves index inside kohonen map
		self.BMUcount = [0]*self.total 	# counts bmu selection for each node
		self.nodeIndex = [] 			# saves index of nodes in order to be able to add or remove nodes
		
	def train(self, train_vector=[[]], iterations=100, continue_training=False):
		''' this is my version of the SOM algorithm. '''
		train_vector = np.asarray(train_vector)
		
		if not continue_training: self.initializeNodes(train_vector)	
		self.resetBmuCount()
		
		for current_iteration in range(1, iterations+1):
			sys.stdout.write("\rTraining Iteration: " + str(current_iteration) + "/" + str(iterations))
			sys.stdout.flush()
			
			current_radius = self.getRadius(current_iteration, iterations)
			current_learning_rate = self.getLearningRate(current_iteration, iterations)
	
			for j in range(int(len(train_vector)/1)):
				best = self.best_match(train_vector[j])
				self.raiseBmuCount(self.getIndex(best[0], best[1]))
				neighborhood = self.find_neighborhood(best, current_radius)
				
				for [neighbourhood_index, dist] in neighborhood:
					influence = self.getKernel(dist, current_radius, current_iteration)
					current_data = self.getNodeVector(neighbourhood_index)
					update = influence * current_learning_rate * (train_vector[j][:] - current_data)
					self.addToNodeVector(neighbourhood_index, update)

		sys.stdout.write("\n")
	
	## node access methods for a rectangular grid
	def getIndex(self, i, j):
		return j + i*self.width
	# set node to a specific array
	def setNode(self, index, array):
		self.nodes[index,:] = array	
	# get node array
	def getNodeVector(self, index):
		return self.nodes[index,:]
	# update node vector
	def addToNodeVector(self, index, array):
		self.nodes[index,:] += array

	## parameter functions
	def getKernel(self, distance, radius, iteration):
		return exp( -1.0 * abs(distance) / (2*radius*iteration) )		
	def getRadius(self, iteration, max_iterations):
		return self.radius * exp(-1.0*log(self.radius)*iteration/max_iterations)
	def getLearningRate(self, iteration, max_iterations):
		return self.learning_rate * exp(-1.0*log(self.radius)*iteration/max_iterations)
    
	## distance measures
	def eucl_distance(self, a, b):
		axis = len(np.shape(a))-1
		return np.sqrt(((a - b)**2).sum(axis=axis))
	def L1_distance(self, r1, r2):
		axis = len(np.shape(r1))-1
		return (abs(r1 - r2)).sum(axis=axis)
		
	# Returns location of best match, uses Euclidean distance
	def best_match(self, target_FV): 
		distances = self.L1_distance(self.nodes, target_FV)
		bestIndex = np.argmin(distances)
		bestkey = self.nodeDict[bestIndex]
		return bestkey
		
	# Returns a list of points which live within 'dist' of 'pt'
	# Uses the Chessboard distance; pt is (row, column)
	def find_neighborhood(self, pt, dist):
		dist = int(dist)
		min_y = max(int(pt[0] - dist), 0)
		max_y = min(int(pt[0] + dist + 1), self.height)
		min_x = max(int(pt[1] - dist), 0)
		max_x = min(int(pt[1] + dist + 1), self.width)
		neighbors = []
		for y in range(min_y, max_y):
			for x in range(min_x, max_x):
				# manhattan distance 
				#dist = (abs(y-pt[0])**2 + abs(x-pt[1])**2)**0.5
				#euclidean distance
				dist = np.sqrt(abs(y-pt[0])**2 + abs(x-pt[1])**2)
				neighbors.append([ self.getIndex(y,x), dist ])
		return neighbors
		
	## initializes nodes as numpy arrays using FV_ranges					
	def initializeNodes(self, train_vector=[[]]):
		self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
		self.nodeDict = [[0,0]]*self.total
		for i in range(self.height):
			for j in range(self.width):
				index = self.getIndex(i,j)
				self.nodeDict[index] = [i, j]			
		self.nodeIndex = dict()
		for i in range(self.total):
			self.nodeIndex[i] = i
		
		if self.FV_ranges == 'random':
			''' random vectors with values within the bounding box of the data '''
			minmax = getBoundingBox(train_vector)
			self.nodes = np.asarray( map( lambda x: np.random.uniform(np.min(minmax),np.max(minmax), (self.FV_size,)) , self.nodes ) )
		elif self.FV_ranges == 'xy_box':
			''' regular grid on the x-y plane. This functionality should be used on PCA-transformed data! '''
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = getBoundingBox(train_vector)
			hh = 1.0/float(self.height-1) if (self.height > 1) else 1
			hw = 1.0/float(self.width-1) if (self.width > 1) else 1
			for i in range(self.height):
				for j in range(self.width):
					index = self.getIndex(i,j)
					self.nodes[index, 0] = minmax[0,0] + i*hh*abs(minmax[0,1]-minmax[0,0])
					self.nodes[index, 1] = minmax[1,0] + j*hw*abs(minmax[1,1]-minmax[1,0])			
		elif self.FV_ranges == '1d_line':
			''' regular line on the x-y plane. '''
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = getBoundingBox(train_vector)
			dimension = max(self.height, self.width)
			h = 1.0/float(dimension-1) if (dimension > 1) else 1
			for i in range(dimension):
					index = self.getIndex(i,0) if self.width < self.height else self.getIndex(0,i)
					self.setNode(index, minmax[:,0] + i*h*np.abs(minmax[:,0]-minmax[:,1]))
		elif len(self.FV_ranges) == 1:
			''' random vectors with values within the range of [[min, max]] '''
			range0, range1 = self.FV_ranges[0][0], self.FV_ranges[0][1]
			self.nodes = np.asarray( map( lambda x: np.random.uniform(range0,range1,(self.FV_size,)) , self.nodes ) )
		else:
			''' random vectors with values within the bounding box of the data '''
			minmax = getBoundingBox(train_vector)
			self.nodes = np.asarray( map( lambda x: np.random.uniform(np.min(minmax),np.max(minmax), (self.FV_size,)) , self.nodes ) )
		
	## methods for counting the Best Matching Unit Index
	def resetBmuCount(self):
		map( lambda x : 0, self.BMUcount )
	def raiseBmuCount(self, index):
		self.BMUcount[index] += 1
	# do this before getting Sampled Data! This removes just from the dictionary
	def removeBmuNodes(self, threshold):
		for n in dict(self.nodeIndex):
			index = self.nodeIndex[n]
			if self.BMUcount[index] <= threshold:
				self.nodeIndex.pop(index)
	
	## obtain the relevant som results 
	def getSampledData(self):
		idcs, nodes, bmus = [], [], []
		for i in self.nodeIndex:
			idcs.append(self.nodeDict[i])
			nodes.append(self.nodes[self.nodeIndex[i]])
			bmus.append(self.BMUcount[self.nodeIndex[i]])
		idcs = np.asarray(idcs)
		nodes = np.asarray(nodes)
		bmus = np.asarray(bmus)
		return idcs, nodes, bmus
	
	## Save the SOM's similarity mask. The darker the area the more rapid the change.
	def build_distance_mask(self):
		tmp_nodes = np.zeros((self.height, self.width), float)
		for r in range(self.height):
			for c in range(self.width):
				neighborhood = self.find_neighborhood((r,c), 1)
				for [n, dist] in neighborhood:
					tmp_nodes[r,c] += self.eucl_distance(self.getNodeVector(self.getIndex(r,c)), self.getNodeVector(n))
		return tmp_nodes
	def save_similarity_mask(self, filename, path="."):
		tmp_nodes = self.build_distance_mask()
		tmp_nodes -= tmp_nodes.min()
		tmp_nodes *= 255 / tmp_nodes.max()
		tmp_nodes = 255 - tmp_nodes
		img = Image.new("L", (self.width, self.height))
		for r in range(self.height):
			for c in range(self.width):
				img.putpixel((c,r), tmp_nodes[r,c])
		img = img.resize((self.width*10,self.height*10),Image.NEAREST)
		img.save(path + "/" + filename + ".png")
################################################################################################
