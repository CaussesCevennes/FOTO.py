# -*- coding:utf-8 -*-

"""
FOTO.py, Fourier Transform Textural Ordination implementation in Python
Author : domlysz, Entente Interdépartementale Causses et Cévennes
Contact : domlysz@gmail.com, observatoire@causses-et-etcevennes.fr
Date : february 2017
Version : WORK IN PROGRESS
License : GPL
Python 3

FOTO (Fourier Transform Textural Ordination)

- Process the input raster by blocks (windows).
- For each block compute Fourier transformation and radial average spectrum.
  Each column of the resulting spectrum contains the portions of the block variance explained by a given frequency.
- Stack all block spectrum into a general array. The number of rows will match the number of blocks whereas number of columns
  will be equals to the chosen number of sampled frequencies 
- Submit the resulting table to a principal component analysis
- Keep the 3 principal components as intensity value to re-build a RGB composite image matching the initial block grid

https://hal.archives-ouvertes.fr/hal-00090693
https://github.com/khufkens/foto

"""




import math, os, sys, time

from threading import Thread, Lock
#from queue import Queue
from multiprocessing import Process, JoinableQueue as Queue, cpu_count

import numpy as np
from osgeo import gdal

try:
	from matplotlib import pyplot as plt
	MPL = True
except:
	MPL = False




class FOTO():

	def __init__(self, inPath, blockSize, method='BLOCK', maxSample=29):
		'''
		inPath : file name of the input raster
		blockSize : size in pixel of the windows
		method	'BLOCK' : aggregate all pixels of the windows
				'NW' : Moving Window, process like a focal stats (Neighborhood analysis)
		maxSample : maximum number of sampled frequencies that will be used for the PCA
		'''
		self.inPath = inPath

		self.method = method
		if self.method == 'MW' and not (blockSize % 2):
			raise ValueError('In moving window mode, window size must be an odd number')

		self.inDataset = gdal.Open(inPath, gdal.GA_ReadOnly)
		self.inWidth, self.inHeight = self.inDataset.RasterXSize, self.inDataset.RasterYSize
		self.nbBands = self.inDataset.RasterCount
		self.blockSize = blockSize

		if self.method == 'BLOCK':
			#Strategy 1: use all data (will produce incomplete blocks)
			#self.nbBlkx = math.ceil(self.inWidth / self.blockSize)
			#self.nbBlky = math.ceil(self.inHeight / self.blockSize)
			
			#Strategy 2 : use only the n first non troncate blocks
			self.nbBlkx = math.floor(self.inWidth / self.blockSize)
			self.nbBlky = math.floor(self.inHeight / self.blockSize)
			#in this case adjust input dimensions
			self.inWidth = self.blockSize * self.nbBlkx
			self.inHeight = self.blockSize * self.nbBlky
			
			self.nbBlock =  self.nbBlkx * self.nbBlky

		elif self.method == 'MW':
			self.nbBlkx, self.nbBlky = self.inWidth, self.inHeight
			self.nbBlock = self.inWidth * self.inHeight

		#Get the optimal number of sampled frequencies N/2 (Nyquist frequency)
		self.nbSample = int(self.blockSize/2)
		#we need at least 3 components
		if self.nbSample < 3:
			self.nbSample = 3
		#limit the number of spectral values to the first most useful ones
		if self.nbSample > maxSample:
			self.nbSample = maxSample

		#Init some props
		self.score = None #the output of the PCA
		self.RGB = None #the RGB matrix based on score data

	def __del__(self):
		# properly close gdal ressources
		self.inDataset = None

	def _printProgress(self, i, t):
		pourcent = round(((i+1)*100)/t)
		if pourcent in list(range(0, 110, 10)) and pourcent != self.progress:
			self.progress = pourcent
			if pourcent == 100:
				print(str(pourcent)+'%')
			else:
				print(str(pourcent), end="%, ")
		sys.stdout.flush() #we need to flush or it won't print anything until after the loop has finished



	def _readRaster(self, nbConsumer=0):
		'''Read raster by block and seed a queue.
		Must be executed by only one thread (gdal is not thread safe)'''
		width, height = self.inWidth, self.inHeight
		blockSize = self.blockSize
		idx = 0

		if self.method == 'BLOCK':
			step = blockSize
			#Loop over blocks and seed the queue
			for y in range(0, height, step):
				if y + blockSize < height:
					ysize = blockSize
				else:
					ysize = height - y
				for x in range(0, width, step):
					if x + blockSize < width:
						xsize = blockSize
					else:
						xsize = width - x

					data = self.inDataset.ReadAsArray(x, y, xsize, ysize)
					self.jobs.put( (idx, data) )
					idx += 1

		elif self.method == 'MW': #MOVING WINDOW
			step = 1
			#Loop over all pixels
			offset = int((blockSize - 1) / 2) #blocksize must be an odd number
			#for each pixel, compute indices of the window (all included)
			for y in range(0, height, step):
				y1 = y - (offset)
				y2 = y + offset
				#adjust the indices
				if y1 < 0: y1 = 0
				if y2 > height-1: y2 = height-1 #px indices start from zero
				ysize = (y2 - y1) + 1
				for x in range(0, width, step):
					x1 = x - (offset)
					x2 = x + offset
					if x1 < 0: x1 = 0
					if x2 > width-1: x2 = width-1
					xsize = (x2 - x1) + 1
					data = self.inDataset.ReadAsArray(x1, y1, xsize, ysize)
					self.jobs.put( (idx, data) )
					idx += 1

		#inject a sentinel to flag the end of the queue (one for each consumer)
		for i in range(nbConsumer):
			self.jobs.put(None)


	def run(self, outPathRGB=False, plot=False):
		'''
		Compute FOTO
		outPathRGB = output path for writing resulting FOTO RGB raster
		The processs is distributed across 3 concurents threads
		- One thread for reading raster data for each windows and seed a job queue
		- A multicore task that process the queue to compute Fourier transformation and radial spectrum,
		  results are then put into another queue
		- The third thread gather results from the second queue to build the respectra table
		  and optionnaly send a data batch to an incremental PCA function
		'''
		t0 = time.time()
		self.progress = 0

		print('Processing FOTO on ' + self.inPath)
		print('Input size is ' +str(self.inWidth)+' x '+str(self.inHeight)+' x '+str(self.nbBands))
		print('Window size is ' + str(self.blockSize))
		print('Number of windows is ' + str(self.nbBlock) + ' (' + str(self.nbBlkx) + ' x ' + str(self.nbBlky) + ')')

		#Init numpy memory mapped array
		self.rSpectra = np.zeros((self.nbBlock, self.nbSample), np.float)
		#cache = os.path.dirname(self.inPath) + os.sep + "rspectra.dat"
		#self.rSpectra = np.memmap(cache, dtype='float32', mode='w+', shape=(self.nbBlock, self.nbSample))#init with zero

		print('Compute radial spectrum for each blocks...')
		#Init a queue to store raster block data, limit the size to not overflow memory (reading is faster than compute rspectra)
		self.jobs = Queue(maxsize=1000)
		#Init a queue to store radial spectra
		self.results = Queue()

		#Get number of process according to number of cpu cores
		self.nbProcess = cpu_count() 
		#If there is only one core available, use multithreading instead of multiprocessing
		#However, cpu-bound calculations will not take a lot of advantage from multithreading
		if self.nbProcess == 1:
			noMultiCore = True
			self.nbProcess = 4
		else:
			noMultiCore = False
		
		#Launch raster reader
		reader = Thread(target=self._readRaster, args=(self.nbProcess,))
		reader.setDaemon(True)
		reader.start()
		
		#Launch radial spectra calculation
		workers = []
		for i in range(self.nbProcess):
			if noMultiCore:
				worker = Thread(target=_calcRspectra, args=(self.jobs, self.results, self.blockSize, self.nbSample))
			else:
				worker = Process(target=_calcRspectra, args=(self.jobs, self.results, self.blockSize, self.nbSample))
			workers.append(worker)
			worker.start()

		#Launch the thread that will gather the data from the result queue
		shepherd = Thread(target=self._gatherData)
		shepherd.setDaemon(True)
		shepherd.start()

		#Wait
		shepherd.join()

		#Ordination through Principale Component Analysis (PCA)
		print('Compute PCA...')
		self.score = PCA(self.rSpectra, 3)

		#Reshape score table to get the RGB matrix
		self.RGB = np.expand_dims(self.score, axis=1)
		self.RGB = self.RGB.reshape((self.nbBlky, self.nbBlkx, 3))
		#self.RGB = np.rollaxis(self.RGB, 0, 3) # because first axis is band index
		#print(self.RGB.shape) # (y, x, b)
		#print(np.shares_memory(self.score, self.RGB)) #make sure the reshape does not trigger a copy

		#Normalize/scale values between 0 and 1
		#(change will also affect self.score because the 2 arrays share the same memory datablock)
		r, g, b = self.RGB[:,:,0], self.RGB[:,:,1], self.RGB[:,:,2]
		#keep correct PCA center (used for ploting pca)
		self.centerx = scale(0, r.min(), r.max(), 0, 1)
		self.centery = scale(0, g.min(), g.max(), 0, 1)
		self.RGB[:,:,0] = npScale(r, 0, 1)
		self.RGB[:,:,1] = npScale(g, 0, 1)
		self.RGB[:,:,2] = npScale(b, 0, 1)
		
		#Write output RGB raster
		if outPathRGB:
			self.writeRGB(outPathRGB)

		t = time.time() - t0
		print('Build in %f seconds' % t)

		#plot
		if plot:
			self.plotPCA()


	def _gatherData(self):
		'''Process results queue to build rspectra table in the right order'''
		for i in range(self.nbBlock):
			idx, rs = self.results.get()
			self.rSpectra[idx,:] = rs
			self._printProgress(i, self.nbBlock)
			
			#TODO
			#Send batch of data to an incremental PCA


	def writeRGB(self, outPath):
		'''Write FOTO results to a new RGB geotiff'''

		print('Write output raster ' + outPath)
		
		if self.RGB is None:
			raise ValueError('FOTO not yet computed')
	
		#build output dataset
		if os.path.isfile(outPath):
				os.remove(outPath)
		driver = gdal.GetDriverByName("GTiff")
		outNbBand = 3
		outDtype = gdal.GetDataTypeByName('Float32')
		outDataset = driver.Create(outPath, self.nbBlkx, self.nbBlky, outNbBand, outDtype)#, ['TFW=YES', 'COMPRESS=LZW'])
		if self.method == 'BLOCK':
			topleftx, pxsizex, rotx, toplefty, roty, pxsizey = self.inDataset.GetGeoTransform()
			outResx = pxsizex * self.blockSize
			outResy = pxsizey * self.blockSize
			outDataset.SetGeoTransform( (topleftx, outResx, rotx, toplefty, roty, outResy) )
		elif self.method == 'MW':
			outDataset.SetGeoTransform(self.inDataset.GetGeoTransform())
		outDataset.SetProjection(self.inDataset.GetProjection())

		'''
		for i in range(outNbBand):
			outDataset.GetRasterBand(i+1).WriteArray(self.RGB[:,:,i])
		'''
		#write by block to avoid memory overflow
		blockSize = 512
		width, height = self.nbBlkx, self.nbBlky
		#Loop over blocks
		for j in range(0, height, blockSize):
			if j + blockSize < height:
				yBlockSize = blockSize
			else:
				yBlockSize = height - j
			for i in range(0, width, blockSize):
				if i + blockSize < width:
					xBlockSize = blockSize
				else:
					xBlockSize = width - i
				
				for b in range(outNbBand):
					outBand = outDataset.GetRasterBand(b+1)
					data = self.RGB[j:j+yBlockSize, i:i+xBlockSize, b]
					outBand.WriteArray(data, i, j)

		# Close
		outDataset = None #needed to validate the modifications



	def plotPCA(self):
		'''plot result of FOTO process : PCA, map and histogram'''

		if not MPL:
			print("Please install Matplotlib to enable ploting")
			return

		if self.RGB is None:
			raise ValueError('FOTO not yet computed')

		fig = plt.figure(figsize=(8, 6))
		fig.subplots_adjust(wspace = 0.5, hspace = 0.5)

		#Plot ACP
		with plt.style.context('seaborn-whitegrid'):
			plt.subplot(2,1,1) #Split the figure into 2 lines and 1 column and get the fist position
			plt.axhline(y=self.centerx, linewidth=1., linestyle='dashed', color="black")
			plt.axvline(x=self.centery, linewidth=1., linestyle='dashed', color="black")
			plt.xlabel('Principal Component 1')
			plt.ylabel('Principal Component 2')

			data = self.score
			plt.scatter(data[:,0], data[:,1], s=4, c=data) #warn color must be normalized from 0 to 1

		#Plot FOTO RGB map
		plt.subplot(2,2,3) #Split the figure into 2 lines and 2 columns and get the third position
		plt.axis("off")
		img = plt.imshow(self.RGB)
		plt.colorbar()

		#Plot map histogram
		plt.subplot(2,2,4)
		plt.hist(self.RGB[:,:,0].ravel(), bins='auto', color=[1,0,0,0.75], histtype='step')
		plt.hist(self.RGB[:,:,1].ravel(), bins='auto', color=[0,1,0,0.75], histtype='step')
		plt.hist(self.RGB[:,:,2].ravel(), bins='auto', color=[0,0,1,0.75], histtype='step')

		plt.show()



#due to pickling limitations, most of the case multiprocessing module cannot deal with functions
#that are members of a class. So we must define the worker outside the class
def _calcRspectra(inQueue, outQueue, blockSize, nbSample):
	'''Worker that process the queue to compute radial spectra'''

	while True:

		#Get a job into the queue
		job = inQueue.get()
		if job is None: #catch sentinel
			break

		idx, data = job

		if len(data.shape) == 3: #multiband
			nbBand, xBlockSize, yBlockSize = data.shape
		else:
			xBlockSize, yBlockSize = data.shape

		#filter incomplete block
		if xBlockSize == yBlockSize == blockSize:
			#if needed compute the mean of all bands
			if len(data.shape) == 3: #multiband
				data = np.mean(data, 0)
			rs = rspectrum(data)
			#only use the first useful harmonics of the r-spectrum
			rs = rs[0:nbSample]
		else:
			#let this line to zeros
			rs = np.zeros((1, nbSample), np.float)

		#put to the result queue
		outQueue.put( (idx, rs))

		#flag it's done
		inQueue.task_done()


##################################
def scale(inVal, inMin, inMax, outMin, outMax):
	'''simple linear strech function'''
	return (inVal - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

def npScale(a, outMin, outMax):
	'''simple linear strech function for numpy array'''
	inMin, inMax = a.min(), a.max()
	return (a - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

#http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
def azimuthalAverage(data, center=None):
	'''
	Calculate the azimuthally averaged radial profile.
	data - The 2D numpy array
	center - The [x,y] pixel coordinates used as the center. The default is
			 None, which then uses the center of the data
	'''
	#Get indices
	y, x = np.indices((data.shape))
	#center
	if not center:
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
	#Get radii as integer
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)
	#sums for each radius bin
	tbin = np.bincount(r.ravel(), data.ravel())
	#number of radius bin
	nr = np.bincount(r.ravel())
	#average
	radialprofile = tbin / nr
	return radialprofile 

def rspectrum(data):
	'''Compute radial spectrum'''
	#check if the block is square or not
	w, h = data.shape[1], data.shape[0]
	#Fast Fourier Transform (FFT) in 2 dims
	ft = np.fft.fft2(data)
	#center fft
	ft = np.fft.fftshift(ft)
	# Calculate a 2D Power Spectrum Density
	psd = np.abs(ft)**2
	# Calculate the azimuthally averaged 1D power spectrum (also called radial spectrum)
	rspect = azimuthalAverage(psd)
	return rspect


def PCA(data, k, svd=False):
	'''Simple Principal Component Analysis implementation to avoid sklearn dependency'''
	#replace nodata and inf values
	data = np.nan_to_num(data)
	#center along column by substracting the mean
	data -= data.mean(axis=0)
	#standardize by dividing with the standard deviation
	data /= data.std(axis=0)
	if svd:
		# Singular Vector Decomposition (SVD)
		u,s,v = np.linalg.svd(data.T) #u == sorted eigvectors
		#get k first components
		u = u[:,:k]
		# Return reduced repr
		return np.dot(data, u)
	else:
		#get normalized correlation matrix...
		#c = np.corrcoef(data.T)
		#...or covariance matrix
		c = np.cov(data.T)
		#get the eigenvalues/eigenvectors
		eigVal, eigVec = np.linalg.eig(c)
		#get sort index of eigenvalues in ascending order
		idx = np.argsort(eigVal)[::-1]
		# sorting eigenvalues
		eigVal = eigVal[idx]
		#sorting the eigenvectors according to the sorted eigenvalues
		eigVec = eigVec[:,idx]
		#cutting some PCs if needed
		eigVec = eigVec[:,:k]
		#return projection of the data in the new space
		return np.dot(data, eigVec)

