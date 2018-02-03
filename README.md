Self Organizing Maps
--------------------
	
This piece of code implements the Self Organizing Maps (SOM) algorithm using numpy.

Using the SOM on different data, one is interested in the similarity mask.
It can show clusters in data. The darker an area is the further away are the feature vectors of the SOM from each other.
By applying SOM.remove_bmu_nodes(threshold) one can also remove interpolating nodes. 

### Contents: ### 
* README	 this read me file
* sompy.py contains the SOM class
* somtest.py contains an example which creates the SOM for 8 base colors.

### Dependencies: ### 
Built against the Python 2.7 on Linux.  Python 3 compatibility unknown.  
No OS specific code used, untested on other operating systems.

* python-numpy
* Saving images (like the similarity mask) requires Python Imaging Library (PIL)

