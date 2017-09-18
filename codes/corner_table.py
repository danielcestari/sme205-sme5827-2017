
import sys
import numpy as np
import matplotlib.pyplot as plt
from grid2vtk import grid2vtk

def compute_corner_table(vertices, faces, oriented=True):
	"""
###
# Create the corner table given the vertices and faces matrices
### 
# vertices:		Matrix. The vertices matrix, coordinates (x,y,z) for every point, vertice
# faces:		Matrix. The faces, in this case triangles, matrix in the VTK format but as python matrix, so the first vertex of the first face is indexed as faces[0,0]
# oriented:		Boolean. If True consider that the faces are all anti-clockwise oriented
###
# return the corner table
	"""
	cn_table_length = 3*len(faces) * 8
	cn_table = np.ndarray(shape=(cn_table_length, 8), dtype=int, buffer=np.array([-1]*cn_table_lenght))

	# create vertex hash
	# a structure used for, given a vertex retrieve the associated corners
	vt_hash = [[]] * len(vertices)

	# the columns of the corner table are:
	# | corner | vertice | triangle | next | previous | opposite | left | right |
	
	# first construct all corner with its vertices, faces, next and previous corners
#	for i in range(1, cn_table_length+1):
	i = 0
	for j in range(len(faces)):
		i = j*3
		
		# which face am I now ?
#		j = 10 # need to compute this index
		fj = faces[j]
		
		ci = cn_table[i,:]
		# assign the corner number, the vertex, and the face for the corner_i
		ci[0], ci[1], ci[2] = i, fj[0], j
		# compute the next and previous corners for the corner_i
		ci[3], ci[4] = fj[1], fj[2]
		# add the corner to the vertex hash
		vt_hash[fj[0]].append(i)
		i += 1
		
		ci = cn_table[i,:]
		# assign the corner number, the vertex, and the face for the corner_i+1
		ci[0], ci[1], ci[2] = i, fj[1], j
		# compute the next and previous corners for the corner_i+1
		ci[3], ci[4] = fj[2], fj[0]
		# add the corner to the vertex hash
		vt_hash[fj[1]].append(i)
		i += 1
		
		ci = cn_table[i,:]
		# assign the corner number, the vertex, and the face for the corner_i+2
		ci[0], ci[1], ci[2] = i, fj[2], j
		# compute the next and previous corners for the corner_i+2
		ci[3], ci[4] = fj[0], fj[1]
		# add the corner to the vertex hash
		vt_hash[fj[2]].append(i)
#		i += 1
		
		
		
		
	
	# then compute the oposite, left and right corners
	for i in range(cn_table_length):
		
		ci = cn_table[i,:]
		ci_n = cn_table[ci[3],:]
		ci_p = cn_table[ci[4],:]
		
		# right corner
		# select c_j \ c_j_vertex == c_i_n_vertex
		# select c_k \ c_k_vertex == c_i_vertex
		# filter to c_k \ c_k_p == c_j
		# THEN c_i_right = c_k_n
		cj = vt_hash[ci_n[1]]
		cks = vt_hash[ci[1]]
		ck_p = set(cn_table[cks, 4])
		ck = ck_p.intersection(cj)
		ci[7] = -1 if not len(ck) else cn_table[ck.pop(), 3]
		

		# left corner
		# select c_j \ c_j_vertex == c_i_p_vertex
		# select c_k \ c_k_vertex == c_i_vertex
		# filter to c_j \ c_j_p == c_k
		# THEN c_i_right = c_j_n
		cjs = vt_hash[ci_p[1]]
		ck = vt_hash[ci[1]]
		cj_p = set(cn_table[cjs, 4])
		cj = cj_p.intersection(ck)
		ci[6] = -1 if not len(cj) else cn_table[cj.pop(), 3]


	for i in range(cn_table_length):
		
		# opposite corner
		# corner_i => next => right
		ci = cn_table[i,:]
		cn = cn_table[ci[3],:]
		ci[5] = cn[7]
	
	vt_hash = None
	
	return cn_table
