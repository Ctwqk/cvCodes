My name: Taiwei Cui

System: Ubuntu 22.04.3
IDE: terminal and several text editors, cmake and make
time travel days usage: 3

program1: establishCsv
usage: ./establishCsv <directory of images> <directory of csv files> <name of feature>

program2: imgMatch
usage: ./imgMatch <path of target image> <directory of csv files> <name of feature> <number of output images>


name of features: 
	7x7central: retrive the central 7x7 part of the image
	gabor: the histogram of gabor filtered image
	dnn: the DNN embedding
	mycustom: the custom design, which is a concatenation of DNN embedding and ROIs of low frequency pass filtered image
	histogram: the 2-D color histogram of image
	multihist: multi-histogram, which is concatenation of 5 ROIs of image
	texturecolor: texture & color
