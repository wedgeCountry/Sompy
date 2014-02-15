from sompy import *

if __name__ == "__main__":

	print "Initialization..."
	colors = [[0, 0, 0], [255, 255, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [0, 0, 255]]
	#transform = sklearn.decomposition.PCA()
	#colors = transform.fit_transform(colors)
	import time
	t0 = time.time()
	width = 20
	height = 20
	iterations = 400
	color_som = SOM(width=width,height=height,FV_size=np.shape(colors)[1],learning_rate=1.0, FV_ranges='xy_box') 
	
	print "Training colors..."
	r = color_som.train(iterations=iterations, train_vector=colors, num_samples = 2+0*len(colors), residual = True)
	
	color_som.save_similarity_mask("test_sim")
	if len(r) > 1:
		print 'Residual:', r[-1]
	t1 = time.time()
	print 'time: %0.2f seconds' %(t1-t0)
	
	print "Saving Image: sompy_test_colors.png..."	
	try:
		img = Image.new("RGB", (width, height))
		for r in range(height):
			for c in range(width):
				
				data = color_som.nodes[color_som.getIndex(r,c)]
	#			data = transform.inverse_transform(data)
				img.putpixel((c,r), (int(data[0]), int(data[1]), int(data[2])))
		img = img.resize((width*10, height*10),Image.NEAREST)
		img.save("sompy_test_colors.png")
	except:
		print "Error saving the image, do you have PIL (Python Imaging Library) installed?"
	print "Saving Image: sompy_original_colors.png..."	
	colors = np.asarray(colors)
	img = Image.new("RGB", (4, 2))
	img.putpixel((0,0),  (0, 0, 0))
	img.putpixel((1,0),  (255, 255, 255))
	img.putpixel((2,0),  (0, 255, 0))
	img.putpixel((3,0),  (0, 255, 255))
	img.putpixel((0,1), (255, 0, 0))
	img.putpixel((1,1), (255, 0, 255))
	img.putpixel((2,1), (255, 255, 0))
	img.putpixel((3,1),  (0, 0, 255))
	img = img.resize((width*10, height*10),Image.NEAREST)
	img.save("sompy_original_colors.png")
	
