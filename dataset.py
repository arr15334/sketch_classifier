from collections import namedtuple

V2 = namedtuple('Point2D', ['x','y'])

import numpy as np
import matplotlib.pyplot as plt

def filter_ds(x):
	x_filtered = []
	for i in range(len(x)):
		if (x[i].shape == (784,)):
			x_filtered.append(x[i])
	return np.array(x_filtered)

x_circles = np.load('./Datasets/Circle/full_numpy_bitmap_circle.npy')[25001:30001]
y_circles = [[.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01] for i in range(len(x_circles))]

x_squares = np.load('./Datasets/Square/full_numpy_bitmap_square.npy')[25001:30001]
y_squares = [[0.01,.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01] for i in range(len(x_squares))]

x_triangles = np.load('./Datasets/Triangle/full_numpy_bitmap_triangle.npy')[25001:30001]
y_triangles = [[0.01,0.01,.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01] for i in range(len(x_triangles))]

x_trees = np.load('./Datasets/Tree/full_numpy_bitmap_tree.npy')[25001:30001]
y_trees = [[0.01,0.01,0.01,.99,0.01,0.01,0.01,0.01,0.01,0.01] for i in range(len(x_trees))]

x_smiley = np.load('./Datasets/smiley_face/full_numpy_bitmap_smiley.npy')[25001:30001]
y_smiley = [[0.01,0.01,0.01,0.01,.99,0.01,0.01,0.01,0.01,0.01] for i in range(len(x_smiley))]

x_house = np.load('./Datasets/House/full_numpy_bitmap_house.npy')[25001:30001]
y_house = [[0.01,0.01,0.01,0.01,0.01,.99,0.01,0.01,0.01,0.01] for i in range(len(x_house))]

x_mickey = np.load('./Datasets/mickey.npy')
y_mickey = [[0.01,0.01,0.01,0.01,0.01,0.01,.99,0.01,0.01,0.01] for i in range(len(x_mickey))]

x_question = np.load('./Datasets/question_mark.npy')
x_question = filter_ds(x_question)
y_question = [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,.99,0.01,0.01] for i in range(len(x_question))]

x_sad = np.load('./Datasets/sad_face.npy')
x_sad = filter_ds(x_sad)
y_sad = [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,.99,0.01] for i in range(len(x_sad))]

x_egg = np.load('./Datasets/eggs.npy')
x_egg = filter_ds(x_egg)
y_egg = [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,.99] for i in range(len(x_egg))]

x_mickey[x_mickey == 0],x_mickey[x_mickey==255] = 230,10
x_sad[x_sad == 0],x_sad[x_sad==255] = 230,10
x_egg[x_egg == 0],x_egg[x_egg==255] = 230,10
x_question[x_question == 0],x_question[x_question==255] = 230,10

x_mickey[x_mickey == 230] = 255
x_sad[x_sad == 230] = 255
x_egg[x_egg == 230] = 255
x_question[x_question == 230] = 255

x_mickey[x_mickey == 10] = 0
x_sad[x_sad == 10] = 0
x_egg[x_egg == 10] = 0
x_question[x_question == 10] = 0

f = 0.99 / 255
x_circles = (np.asfarray(x_circles) * f + 0.01)
x_squares = (np.asfarray(x_squares) * f + 0.01)
x_triangles = (np.asfarray(x_triangles) * f + 0.01)
x_trees = (np.asfarray(x_trees) * f + 0.01)
x_smiley= (np.asfarray(x_smiley) * f + 0.01)
x_house = (np.asfarray(x_house) * f + 0.01)
x_mickey = (np.asfarray(x_mickey) * f + 0.01)
x_question = (np.asfarray(x_question) * f + 0.01)
x_sad = (np.asfarray(x_sad) * f + 0.01)
x_egg = (np.asfarray(x_egg) * f + 0.01)

circles = V2(x_circles, y_circles)
squares = V2(x_squares, y_squares)
triangles = V2(x_triangles, y_triangles)
trees = V2(x_trees, y_trees)
smileys = V2(x_smiley, y_smiley)
house = V2(x_house, y_house)
mickey = V2(x_mickey, y_mickey)
question = V2(x_question, y_question)
sad = V2(x_sad, y_sad)
egg = V2(x_egg, y_egg)

full_dataset_x = np.concatenate((x_circles,x_squares))
full_dataset_x = np.concatenate((full_dataset_x,x_triangles))
full_dataset_x = np.concatenate((full_dataset_x,x_trees))
full_dataset_x = np.concatenate((full_dataset_x,x_smiley))
full_dataset_x = np.concatenate((full_dataset_x,x_house))
full_dataset_x = np.concatenate((full_dataset_x,x_mickey))
full_dataset_x = np.concatenate((full_dataset_x,x_question))
full_dataset_x = np.concatenate((full_dataset_x,x_sad))
full_dataset_x = np.concatenate((full_dataset_x,x_egg))

full_dataset_y = np.concatenate((y_circles,y_squares))
full_dataset_y = np.concatenate((full_dataset_y,y_triangles))
full_dataset_y = np.concatenate((full_dataset_y,y_trees))
full_dataset_y = np.concatenate((full_dataset_y,y_smiley))
full_dataset_y = np.concatenate((full_dataset_y,y_house))
full_dataset_y = np.concatenate((full_dataset_y,y_mickey))
full_dataset_y = np.concatenate((full_dataset_y,y_question))
full_dataset_y = np.concatenate((full_dataset_y,y_sad))
full_dataset_y = np.concatenate((full_dataset_y,y_egg))

X = full_dataset_x
Y = full_dataset_y
#print(full_dataset[0].shape)

def draw():
 	img = x_sad[3].reshape((28,28))
 	plt.imshow(img, cmap='Greys')
 	plt.show()

#draw()
#X = np.array([x_circles,x_squares,x_triangles,x_trees,x_smiley,x_house,x_mickey,x_question,x_sad,x_egg])
#Y = np.array([y_circles,y_squares,y_triangles,y_trees,y_smiley,y_house,y_mickey,y_question,y_sad,y_egg])

#full_dataset = V2(X,Y)
#np.save('full_dataset', full_dataset)