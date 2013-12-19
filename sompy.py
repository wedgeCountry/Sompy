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

from random import *
import random
from math import *
import sys
from PIL import Image
import scipy.ndimage
import numpy as np


''' Abstract SOM: No specific structure for nodes -> specify in initializeNodes(), call using getNodeVector, addToNodeVector '''
class AbstractSom():
	def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005, FV_ranges=None):
		self.height = height
		self.width = width
		self.total = self.width * self.height
		self.FV_size = FV_size
		self.FV_ranges = FV_ranges
		self.learning_rate = learning_rate
		self.radius = 1 # !! choose > 1
		
	def train(self, train_vector=[[]], iterations=100, continue_training=False):
		
		train_vector = np.asarray(train_vector)
		if continue_training is False:
			self.initializeNodes(train_vector)	
		else: self.initBmuCount()
		
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
					influence = self.getInfluence(dist, current_radius, current_iteration)
					current_data = self.getNodeVector(neighbourhood_index)
					update = influence * current_learning_rate * (train_vector[j][:] - current_data)
					self.addToNodeVector(neighbourhood_index, update)
		sys.stdout.write("\n")

	def build_distance_mask(self):
		tmp_nodes = np.zeros((self.width, self.height), float)
		for r in range(self.height):
			for c in range(self.width):
				neighborhood = self.find_neighborhood((r,c), 1)
				for [n, dist] in neighborhood:
					tmp_nodes[r,c] += self.FV_distance(self.getNodeVector(self.getIndex(r,c)), self.getNodeVector(n))
		return tmp_nodes
		
	# Show smoothness of the SOM.  The darker the area the more rapid the change, generally bad.
	def save_similarity_mask(self, filename, path="."):
		tmp_nodes = self.build_distance_mask()
		#tmp_nodes -= tmp_nodes.min()
		tmp_nodes *= 255 / tmp_nodes.max()
		tmp_nodes = 255 - tmp_nodes
		img = Image.new("L", (self.width, self.height))
		for r in range(self.height):
			for c in range(self.width):
				img.putpixel((c,r), tmp_nodes[r,c])
		img = img.resize((self.width*10,self.height*10),Image.NEAREST)
		img.save(path + "/" + filename + ".png")

########################################################################################################
		
def getBoundingBox(samples):
	samples = np.asarray(samples)
	minmax = np.asarray( [ [0.0]*2 ] * np.shape(samples)[1])
	for i in range(np.shape(samples)[1]):
		minmax[i,0] = min(samples[:,i])
		minmax[i,1] = max(samples[:,i])
	return minmax

########################################################################################################

''' Implementation of AbstractSom using numpy arrays. Very fast! '''
class SOM(AbstractSom):
	def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.05, FV_ranges=None):
		AbstractSom.__init__(self, height=height, width=width, FV_size=FV_size, learning_rate=learning_rate, FV_ranges=FV_ranges)
		#TODO: Understand!
		self.radius = (height+width)/3
		self.nodes = [] # saves node data
		self.nodeDict = [] # saves index inside kohonen map
		self.BMUcount = [] # counts bmu selection for each node
		self.nodeIndex = [] # saves index of nodes in order to be able to add or remove nodes
	
	# returns the Euclidean distance between two Feature Vectors
	# FV_1, FV_2 are numpy arrays
	def FV_distance(self, FV_1, FV_2):
		# Euclidean distance
		axis = len(np.shape(FV_1))-1
		dist = (((FV_1 - FV_2)**2).sum(axis=axis))**0.5
		return dist
	# Returns location of best match, uses Euclidean distance
	def best_match(self, target_FV): 
		distances = self.FV_distance(self.nodes, target_FV)
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
				dist = abs(y-pt[0]) + abs(x-pt[1])
				neighbors.append([ self.getIndex(y,x), dist ])
		return neighbors
	
	# get decaying radius	
	def getRadius(self, iteration, max_iterations):
		return self.radius * exp(-1.0*iteration*log(self.radius)/max_iterations)
		
	# get learning rate for each iteration
	def getLearningRate(self, iteration, max_iterations):
		return self.learning_rate * exp(-1.0*iteration*log(self.radius)/max_iterations)

	# get adaption coefficient
	def getInfluence(self, distance, radius, iteration):
			return exp( -1.0 * (distance**2) / (2*radius*iteration) )		
	
	def setNode(self, index, data):
		self.nodes[index,:] = data
		
	def getNodeVector(self, index):
		return self.nodes[index,:]
		
	def addToNodeVector(self, index, array):
		self.nodes[index,:] += array

	def getIndex(self, i, j):
		return j + i*self.width
				
	def initializeNodes(self, train_vector=[[]]):
		self.nodeIndex = [ i for i in range(self.total)]
		self.BMUcount = [0]*self.total
		self.nodeDict = [[0,0]]*self.total
		self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
		for i in range(self.height):
			for j in range(self.width):
				index = self.getIndex(i,j)
				self.nodeDict[index] = [i, j]			

		if not self.FV_ranges:
			''' random vectors with values between 0 and 100 '''
			self.nodes = np.asarray( map( lambda x: np.random.uniform(0,100,(self.FV_size,)) , self.nodes ) )
		elif self.FV_ranges == 'xy_box':
			''' regular grid on the x-y plane. This functionality shoul be used on PCA-transformed data! '''
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = getBoundingBox(train_vector)
			hh, hw = 1.0/float(self.height-1), 1.0/float(self.width-1)
			for i in range(self.height):
				for j in range(self.width):
					index = self.getIndex(i,j)
					self.nodes[index, 0] = minmax[0,0] + i*hh*abs(minmax[0,1]-minmax[0,0])
					self.nodes[index, 1] = minmax[1,0] + j*hw*abs(minmax[1,1]-minmax[1,0])			
		elif self.FV_ranges == 'minmax_box':
			''' random vectors with values within the bounding box of the data '''
			self.nodes = np.asarray([ [0.0]*self.FV_size] * self.total)
			minmax = getBoundingBox(train_vector)
			rand_init = lambda x : np.asarray([ random.uniform(minmax[i][0], minmax[i][1]) for i in range(self.FV_size) ])
			self.nodes = np.asarray( map( rand_init , self.nodes ) )
		elif len(self.FV_ranges) == 1:
			''' random vectors with values within the range of [[min, max]] '''
			range0, range1 = self.FV_ranges[0][0], self.FV_ranges[0][1]
			self.nodes = np.asarray( map( lambda x: np.random.uniform(range0,range1,(self.FV_size,)) , self.nodes ) )
		else:
			''' random vectors with values within the range of [[min_1, max_1] ... [min_n, max_n]] '''
			rand_init = lambda x : np.asarray([ random.uniform(self.FV_ranges[i][0], self.FV_ranges[i][1]) for i in range(self.FV_size) ])
			self.nodes = np.asarray( map( rand_init , self.nodes ) )
		
	def initBmuCount(self):
		map( lambda x : 0, self.BMUcount )
	def raiseBmuCount(self, index):
		self.BMUcount[index] += 1

	# do this before getting Sampled Data! This removes just from the dictionary
	def removeBmuNodes(self, threshold):
		for n in np.copy(self.nodeIndex):
			index = self.nodeIndex[n]
			if self.BMUcount[index] <= threshold:
				self.nodeIndex.pop(index)
				self.nodeDict.pop(index)
				self.BMUcount.pop(index)
				self.nodes.pop(index)
	
	def getSampledData(self):
		return self.nodeDict, self.nodes, self.BMUcount
		
################################################################################################

class Linear_SOM(SOM):
	def __init__(self, height=10, width=10, FV_size=10, learning_rate=1):
		SOM.__init__(self, height=height, width=width, FV_size=FV_size, learning_rate=learning_rate)
	
	# get decaying radius	
	def getRadius(self, iteration, max_iterations):
		return self.radius * ((max_iterations - iteration) / max_iterations)
	# get learning rate for each iteration
	def getLearningRate(self, iteration, max_iterations):
		return self.learning_rate * ((max_iterations - iteration) / max_iterations)
	# get adaption coefficient
	def getInfluence(self, distance, radius, iteration):
		return (-distance + radius) / max(1,radius)



if __name__ == "__main__":

	print "Initialization..."
	colors = [[0, 0, 0], [255, 255, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [0, 0, 255]]
	
	width = 100
	height = 100
	iterations = 500
	color_som = SOM(width=width,height=height,FV_size=np.shape(colors)[1],learning_rate=0.5, FV_ranges='minmax_box') 

	print "Training colors..."
	color_som.train(iterations=iterations, train_vector=colors)
	color_som.save_similarity_mask("test_sim")
	
	print "Saving Image: sompy_test_colors.png..."	
	try:
		img = Image.new("RGB", (width, height))
		for r in range(height):
			for c in range(width):
				
				data = color_som.getNodeVector(color_som.getIndex(r,c))
				
				img.putpixel((c,r), (int(data[0]), int(data[1]), int(data[2])))
		img = img.resize((width*10, height*10),Image.NEAREST)
		img.save("sompy_test_colors.png")
	except:
		print "Error saving the image, do you have PIL (Python Imaging Library) installed?"
		
