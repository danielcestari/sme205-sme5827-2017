# First practical exercice

import sys
from matplotlib import pyplot as plt
import numpy as np


# read the air foil file
def read_airfoil(filename):
	f = open(filename, 'rt')
	return np.array([[float(j) for j in i.split(sep=' ', maxsplit=1)] for i in f.read().splitlines()])



save_file = 'output.txt' if len(sys.argv) < 3 else sys.argv[2]

filename = 'naca012.txt' if len(sys.argv) < 2 else sys.argv[1]
air = read_airfoil(filename)
max_x_af = max(air[:,0])
min_x_af = min(air[:,0])
id_max_x_af = np.where(air[:,0] == max_x_af)[0]
id_min_x_af = np.where(air[:,0] == min_x_af)[0]
max_y_af = max(air[:,1])
min_y_af = min(air[:,1])
id_max_y_af = np.where(air[:,0] == max_y_af)[0]
id_min_y_af = np.where(air[:,0] == min_y_af)[0]


# use as the reference where the frame will be split the rightmost point
id_A_C = id_max_x_af

# first define the whole frame with the air foil in the center of it
# parametrize it, using offset values, like horizontal and vertical 
# spaces between the air foil border
v_offset = 0.5
h_offset = 1.0
max_x, min_x  = max_x_af + h_offset, min_x_af - h_offset
max_y, min_y = max_y_af + v_offset, min_y_af - v_offset


# the number of points to describe the air foil determine the grid resolution
resolution = len(air)
# the file naca012.txt begin tracing the top of the air foil from the right
top_bd = air

# the bottom of the grid should follow the path from B -> H -> G -> F -> E -> D
# according to the specification (PDF)

# first need to total length for this path, the computation is made for each segment
# from A_y to max_y, from H to G, from G to F, from F to E, from E to D
# since A and B are at the same y level, I use the y value from A
# the is happens for C and D
bottom_length = abs(abs(max_y) - abs(air[0, 1])) + 2*(abs(max_x) + abs(min_x)) + (abs(max_y) + abs(min_y)) + abs(abs(min_y) - abs(air[-1, 1]))
bottom_bd = []
"""
print(bottom_length)
print(resolution * (abs(max_y) - abs(air[0, 1])) / bottom_length )
print(resolution * (abs(max_x) + abs(min_x)) / bottom_length ) 
print(resolution * (abs(max_y) + abs(min_y)) / bottom_length ) 
print(resolution * (abs(max_x) + abs(min_x)) / bottom_length ) 
print(resolution - (len(seg_BH) + len(seg_HG) + len(seg_GF) + len(seg_FE))  ) 
print(('last', resolution * abs(abs(air[-1, 1]) - abs(min_y)) / bottom_length ))
#"""
# now compute the points for each segment that compose the bottom border
# B-H
seg_BH = [[max_x, i] for i in np.linspace(start=air[0, 1], stop=max_y, 
			num=int(resolution * (abs(max_y) - abs(air[0, 1])) / bottom_length ))] 
# H-G
seg_HG = [[i, max_y] for i in np.linspace(start=max_x, stop=min_x, 
			num=int(resolution * (abs(max_x) + abs(min_x)) / bottom_length ))] 
# G-F
seg_GF = [[min_x, i] for i in np.linspace(start=max_y, stop=min_y, 
			num=int(resolution * (abs(max_y) + abs(min_y)) / bottom_length ))] 
# F-E
seg_FE = [[i, min_y] for i in np.linspace(start=min_x, stop=max_x, 
			num=int(resolution * (abs(max_x) + abs(min_x)) / bottom_length ))] 
# E-D
seg_ED = [[max_x, i] for i in np.linspace(start=min_y, stop=air[-1, 1], 
			num=(resolution - (len(seg_BH) + len(seg_HG) + len(seg_GF) + len(seg_FE))  ))] 
#			num=int(resolution * abs(abs(air[-1, 1]) - abs(min_y)) / bottom_length ))] 
bottom_bd = np.vstack( (seg_BH, seg_HG, seg_GF, seg_FE, seg_ED) )



# TODO change the resolution for the left and right borders

# the left border
# from A to B
left_bd = np.array( [[i, air[0, 1]] for i in np.linspace(stop=air[0, 0], start=max_x, 
			num=resolution)] )

# the right border
# from C to D
right_bd = np.array( [[i, air[-1, 1]] for i in np.linspace(stop=air[-1, 0], start=max_x, 
			num=resolution)] )

#save_file = './airfoil.txt'
f = open(save_file, 'wt')
borders = [top_bd, bottom_bd, left_bd, right_bd]
for bd in borders:
	f.write('%d\n'%(len(bd)))
	[f.write('%.2f %.2f\n'%(i[0], i[1])) for i in bd]
f.close()

# given a file, read the boundary following the Nonato's convention
# top, bottom, left, right
def read_boundary(filename):
	f = open(filename, 'rt')
	data = [i.split(' ') for i in f.read().splitlines()]
	boundary = []
	i = 0
	while i < len(data):
		if len(data[i]) != 1:
			print('ERROR 1')
			break
		size = int(data[i][0])
		i += 1
		border = data[i:(i+size)]
		boundary.append([])
		for j in border:
			if len(j) != 2:
				print('Error 2')
				break
			boundary[-1].append([float(j[0]), float(j[1])])
		i = i + size 
	return boundary
	
# to plot from the boundary file just read
# [(plt.plot(bd[:,0], bd[:,1])) for bd in swan]; plt.show();
