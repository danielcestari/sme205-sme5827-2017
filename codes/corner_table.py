
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
	cn_table_length = 3*len(faces) +1
	cn_table = np.ndarray(shape=(cn_table_length, 8), dtype=int, buffer=np.array([-1]*cn_table_length*8))

	# create vertex and face hash
	# a structure used for, given a vertex retrieve the associated corners
	vt_hash, fc_hash = [], []
	[vt_hash.append([]) for i in range(len(vertices)+1)]
	[fc_hash.append([]) for i in range(len(faces)+1)]

	# the columns of the corner table are:
	# | corner | vertice | triangle | next | previous | opposite | left | right |
	
	# first construct all corner with its vertices, faces, next and previous corners
#	for i in range(1, cn_table_length+1):
	i = 1
	for j in range(len(faces)):
		i = j*3 +1
		
		# which face am I now ?
#		j = 10 # need to compute this index
		fj = faces[j]
		
		ci = cn_table[i,:]
		print(('i', i))
		print(('j', j+1 ))
		print(('cj', ci))
		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i
		ci[0], ci[1], ci[2] = i, fj[0], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i
#		ci[3], ci[4] = fj[1], fj[2]
		# add the corner to the vertex hash
		vt_hash[fj[0]].append(i)
		i += 1
		print(('vt_hash', vt_hash))
		print(('fc_hash', fc_hash))
		print()
		
		ci = cn_table[i,:]
		print(('i', i))
		print(('j', j+1 ))
		print(('cj', ci))
		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i+1
		ci[0], ci[1], ci[2] = i, fj[1], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i+1
#		ci[3], ci[4] = fj[2], fj[0]
		# add the corner to the vertex hash
		vt_hash[fj[1]].append(i)
		i += 1
		print(('vt_hash', vt_hash))
		print(('fc_hash', fc_hash))
		print()
		
		ci = cn_table[i,:]
		print(('i', i))
		print(('j', j+1 ))
		print(('cj', ci))
		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i+2
		ci[0], ci[1], ci[2] = i, fj[2], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i+2
#		ci[3], ci[4] = fj[0], fj[1]
		# add the corner to the vertex hash
		vt_hash[fj[2]].append(i)
#		i += 1
		print(('vt_hash', vt_hash))
		print(('fc_hash', fc_hash))
		print()
		
	i = 1
	while i < cn_table_length:	
#	for i in range(1, cn_table_length):
		
		# which face am I now ?
#		j = 10 # need to compute this index
		#fj = faces[j-1]
		print(len(cn_table))
		print(cn_table_length)
		print(i)
		ci = cn_table[i,:]
		corners = fc_hash[ci[2]]
		# compute the next and previous corners for the corner_i
		ci[3], ci[4] = corners[1], corners[2]
		i += 1
		ci = cn_table[i,:]
		# compute the next and previous corners for the corner_i+1
		ci[3], ci[4] = corners[2], corners[0]
		i += 1
		ci = cn_table[i,:]
		# compute the next and previous corners for the corner_i+2
		ci[3], ci[4] = corners[0], corners[1]
		i += 1
		
	# JUST GIVE AN EXTRA NEXT ON THESE 3 THAT SOLVE ACCORDING TO THE EXAMPLE	

	# then compute the oposite, left and right corners
	for i in range(1, cn_table_length):
		
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


	for i in range(1, cn_table_length):
		
		# opposite corner
		# corner_i => next => right
		ci = cn_table[i,:]
		cn = cn_table[ci[3],:]
		ci[5] = cn[7]
	
	vt_hash = None
	
	return cn_table



def read_vtk(filename):
	"""
#####
# Read vtk file 
###
# 
###
# Return a tuple with the vertices and the faces
	"""
	import csv
	
	f = open(filename, 'rt')
	txt = f.readlines()
	pts_init = 0
	pts_size = 0
	pts_type = float
	fcs_size = 0
	for i,st in enumerate(txt):
		if(st.find('POINTS') != -1):
			tmp = st.split(' ')
			pts_init = i+1
			pts_size = int(tmp[1])
			pts_type = float if tmp[2].find('float') != -1 else int
			break
	# use the csv to read and separate the columns using space as the separator
	# then convert it to list so the np.array can be used
	pts = np.array( list( csv.reader(txt[pts_init:(pts_init + pts_size)], delimiter=' ') ), dtype=pts_type)
	
	for i,st in enumerate(txt[(pts_init+pts_size):]):
		if(st.find('CELLS') != -1):
			tmp = st.split(' ')
			fcs_init = i+1
			fcs_size = int(tmp[1])
			break
	# use the csv to read and separate the columns using space as the separator
	# then convert it to list so the np.array can be used
	fcs = np.array( list( csv.reader(txt[(pts_init + pts_size + fcs_init):(pts_init + pts_size + fcs_init + fcs_size)], delimiter=' ') ), dtype=int)
	
	return (pts, fcs)
