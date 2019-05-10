import sys, os, time, imageio
import numpy as np
import math, random
#from dataset import X,Y

images_dict = [
	'circle',
	'square',
	'triangle',
	'trees',
	'smiley',
	'house',
	'mickey',
	'question_mark',
	'sad',
	'egg'
]

class Neural_Network(object):
	def __init__(self):
		# Initialize network
		self.INPUT_SIZE = 784 # 28x28
		self.OUTPUT_SIZE = 10 # 10 possible drawings
		self.HIDDEN_NODES = 300 #arbitrary

		self.THETA_1 = np.random.randn(self.HIDDEN_NODES,self.INPUT_SIZE+1)
		self.THETA_2 = np.random.randn(self.OUTPUT_SIZE,self.HIDDEN_NODES+1)

		self.sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
		self.sigmoid_derivate = np.vectorize(lambda x: self.sigmoid(x) * (1 - (x)))

	def feed_forward(self,X):
		r = len(X)
		x = X.reshape(r,1)
		a1 = np.vstack((1,x))
		#a1 = self.sigmoid(z1)

		z2 = np.matmul( self.THETA_1, a1 )
		a2 = self.sigmoid(z2)

		z3 = np.matmul(self.THETA_2, np.vstack((1,a2)))
		a3 = self.sigmoid(z3)

		forward_results	= {'a1': a1, 'z2':z2,'a2': a2, 'z3': z3, 'yh': a3}
		return forward_results

	def feed_forward_m(self, X):
		m,n = X.shape # m , 784
		a1 = np.hstack((np.ones(m).reshape(m,1),X))

		z2 = np.matmul(self.THETA_1, a1.T)
		a2 = self.sigmoid(z2).T
		m2,n2 = a2.shape
		a2 = np.hstack((np.ones(m2).reshape(m2,1), a2))

		z3 = np.matmul(self.THETA_2, a2.T)
		a3 = self.sigmoid(z3).T

		forward_results = {'a1':a1, 'z2': z2, 'a2': a2, 'z3': z3, 'Yh': a3}
		return forward_results

	def backward(self,X, Y):
		m, n = X.shape
		DELTA1 = np.zeros((self.HIDDEN_NODES, self.INPUT_SIZE+1))
		DELTA2 = np.zeros((self.OUTPUT_SIZE, self.HIDDEN_NODES+1))
		for i in range(m):
			forward = self.feed_forward(X[i])
			yh = forward['yh']
			a1 = forward['a1']
			a2 = forward['a2']

			sigma3 = yh - Y[i].reshape((10,1)) 
					 # (r,10 x 10 ,1)   * (r,1)
			sigma2 = (self.THETA_2[:,1:].T @ sigma3) * a2 * (1 - a2)

					# (785, r) x (r,1) * (r,1)
			sigma1 = (self.THETA_1[:,1].T @ sigma2) * a1 * (1 - a1)

					# (r,785) + (r,1) x (1,785)
			DELTA1 = DELTA1 + sigma2 @ a1.T
						
					# (10,r+1) + (10,1) (1, r+1)
			DELTA2 = DELTA2 + sigma3 @ np.vstack((1,a2)).T 

		DELTA1 = DELTA1 / m
		DELTA2 = DELTA2 / m
		return DELTA1,DELTA2

	def gradient_descent(self, tX, tY, alpha, threshold = 0.001, max_iter = 25):
		# TODO
		# obtener deltas
		m,n = tX.shape
		print("m gd",m)
		#alpha_m = np.ones((self.HIDDEN_NODES+1,m)) * alpha
		i = 0
		converged = False
		last_cost = 999999999
		get_cost_derivate_v = np.vectorize(self.get_cost_derivate)
		while i < max_iter and not(converged):
			#Yh = self.feed_forward_m(tX)['Yh']
			#last_cost = self.get_cost_value(Yh, tY)
			# UPDATE WEIGHTS
			DELTA1, DELTA2 = self.backward(tX,tY)
			self.THETA_2 -= alpha * DELTA2 
			self.THETA_1 -= alpha * DELTA1
			if (i % 5 == 0):
				Yh = self.feed_forward_m(tX)['Yh']
				current_cost = self.get_cost_value(Yh, tY)
				current_linear = self.linear_cost_function(Yh, tY)
				print("curr_cost cross_entropy",current_cost)
				print("curr_cost linear",current_linear)
			#if (i > 30):
				#alpha = 0.2
			#converged = abs(last_cost - current_cost) < threshold or (self.get_cost_derivate(Yh,tY) < threshold)
			i += 1

	def linear_cost_function(self, Yh, Y):
		m, _ = Y.shape
		print("Y", Y.shape)
		print("Yh", Yh.shape)
		error = 1/(2*m) * (Yh - Y)**2
		return error.sum()

	def get_cost_value(self, Yh, Y):
		#m = len(y)
		m , _= Y.shape
		#cost = (yh-y.reshape(m,1))**2
		a_m = Y @ np.log(Yh).T
		b_m = (1 -Y ) @ (1 - np.log(Yh)).T
		#a = np.dot(y.reshape((m,1)), np.log(yh).T)
		#b = np.dot(1-y.reshape((m,1)), np.log(1-yh).T)
		cost = - (1/m) * (a_m + b_m)

		return np.linalg.norm(cost)


	def get_cost_derivate(self, Yh, Yt):
		m, _ = Yt.shape
		# y/yh - 1-y / 1 - yh
		c = -1 * ( Yt/Yh - (1-Yt)/(1-Yh))
		return np.linalg.norm(c)

	def get_cost_derivate_m(self, Yh, Y):
		return -1 * (Y/Yh - (1 - Y) / (1 - Yh))

	def evaluate(self, test_x, test_y):
		mx, nx = test_x.shape #(m, 784)
		my, ny = test_y.shape #(m, 10)
		correct = 0

		for i in range(mx):
			yh = self.feed_forward(test_x[i])['yh']
			y = test_y[i].reshape((10,1))

			guess = np.argmax(yh)
			real = np.argmax(y)

			if (guess == real):
				correct += 1
		return correct

	def run(self):
		print("Running")
		while True:
			time.sleep(2)
			files = os.listdir('./test')
			for f in files:
				if f.endswith('.bmp'):
					#self.parse_image('./test/'+f)
					print(f)
					self.predict_image('./test/' + f)
					os.rename('./test/'+f,'./old/'+f)
					print("yelo")

	def run_continuos(self):
		print("Running")
		while True:
			files = os.listdir('./test')
			for f in files:
				if f.endswith('.bmp'):
					#self.parse_image('./test/'+f)
					print(f)
					self.predict_image('./test/' + f)
					time.sleep(4)


	def predict_image(self, path):
		print('Predicting...')
		image = imageio.imread(path, as_gray = True)
		image_array = np.array(image).flatten()
		image_array[image_array == 0],image_array[image_array == 255] = 230,10
		image_array[image_array == 10],image_array[image_array == 230] = 0, 255
		f = 0.99 / 255
		x_image = (np.asfarray(image_array) * f + 0.01)
		#print(x_image)
		feed_res = self.feed_forward(x_image)
		yh = feed_res['yh']
		print("yh np argsor", np.argsort(yh.flatten()))
		sols = np.argsort(yh.flatten())[-3:]
		print(sols)
		print(yh*100)
		#print(self.feedforward(x_image))
		hyp = np.argmax(yh)
		print(hyp*100)
		print("Primero: ", images_dict[sols[2]])
		print("Segundo: ", images_dict[sols[1]])
		print("Tercero: ", images_dict[sols[0]])
		print("Dibujo: " , images_dict[hyp] , ", confianza: ",yh[hyp]*100)

		

# INITIALIZE
nn = Neural_Network()

t1 = np.load('T1_correct.npy')
t2 = np.load('T2_correct.npy')

nn.THETA_1 = t1
nn.THETA_2 = t2

nn.run_continuos()

#nn.run()
# BATCH 1
#all_index = list(zip(X,Y))
#np.random.shuffle(all_index)
#sx, sy = zip(*all_index)
#sx = np.array(sx)
#sy = np.array(sy)

#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))

#nn.gradient_descent(tX = sx[:5000], tY = sy[:5000], alpha = .08)
#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))

# BATCH 2
#all_index = list(zip(X,Y))
#np.random.shuffle(all_index)
#sx, sy = zip(*all_index)
#sx = np.array(sx)
#sy = np.array(sy)

#nn.gradient_descent(tX = sx[5000:10000], tY = sy[5000:10000], alpha = .08)
#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))


#BATCH 3
#all_index = list(zip(X,Y))
#np.random.shuffle(all_index)
#sx, sy = zip(*all_index)
#sx = np.array(sx)
#sy = np.array(sy)

#nn.gradient_descent(tX = sx[10000:15000], tY = sy[10000:15000], alpha = 1.01)
#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))

#BATCH 4
#all_index = list(zip(X,Y))
#np.random.shuffle(all_index)
#sx, sy = zip(*all_index)
#sx = np.array(sx)
#sy = np.array(sy)

#nn.gradient_descent(tX = sx[15000:20000], tY = sy[15000:20000], alpha = 1.02)
#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))

#BATCH 5
#all_index = list(zip(X,Y))
#np.random.shuffle(all_index)
#sx, sy = zip(*all_index)
#sx = np.array(sx)
#sy = np.array(sy)

#nn.gradient_descent(tX = sx[20000:25000], tY = sy[20000:25000], alpha = 0.02)
#print(nn.evaluate(sx[30000:32100], sy[30000:32100]))

# save weights
#np.save('T1_correct.npy', nn.THETA_1)
#np.save('T2_correct.npy', nn.THETA_2)
print("Done")



"""
print("Training started...")
# separate test set from training set
all_index = list(zip(X,Y))
np.random.shuffle(all_index)
#random.shuffle(all_index)
#random.shuffle(all_index)
sx, sy = zip(*all_index)

print("Shuffled 3x")

new_training_set = (sx[:450],sy[:450])
new_test_set = (sx[1000:1100], sy[1000:1100])

#nn.SGD( new_training_set, 5, 100, 2.5, test_set = new_test_set)
nn.run()
"""
#nn.backward(X,Y)
#print("Real", Y[120000])
#print("Predicted", nn.feed_forward(X[120000]))

#nn.run()
