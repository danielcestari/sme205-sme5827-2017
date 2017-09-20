
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
# oriented:		Boolean. If True consider that the faces are all counter-clockwise oriented
###
# return the corner list per vexter and the corner table in a tuple
	"""
	cn_table_length = 3*len(faces) +1
	cn_table = np.ndarray(shape=(cn_table_length, 8), dtype=int, buffer=np.array([-1]*cn_table_length*8))

	# create vertex and face hash
	# a structure used for, given a vertex retrieve the associated corners
	# vt_hash is returned as the list of corners for each vertex
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
#		print(('i', i))
#		print(('j', j+1 ))
#		print(('cj', ci))
#		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i
		ci[0], ci[1], ci[2] = i, fj[0], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i
#		ci[3], ci[4] = fj[1], fj[2]
		# add the corner to the vertex hash
		vt_hash[fj[0]].append(i)
		i += 1
#		print(('vt_hash', vt_hash))
#		print(('fc_hash', fc_hash))
#		print()
		
		ci = cn_table[i,:]
#		print(('i', i))
#		print(('j', j+1 ))
#		print(('cj', ci))
#		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i+1
		ci[0], ci[1], ci[2] = i, fj[1], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i+1
#		ci[3], ci[4] = fj[2], fj[0]
		# add the corner to the vertex hash
		vt_hash[fj[1]].append(i)
		i += 1
#		print(('vt_hash', vt_hash))
#		print(('fc_hash', fc_hash))
#		print()
		
		ci = cn_table[i,:]
#		print(('i', i))
#		print(('j', j+1 ))
#		print(('cj', ci))
#		print(('fj', fj))
		# assign the corner number, the vertex, and the face for the corner_i+2
		ci[0], ci[1], ci[2] = i, fj[2], j+1
		fc_hash[j+1].append( i )
		# compute the next and previous corners for the corner_i+2
#		ci[3], ci[4] = fj[0], fj[1]
		# add the corner to the vertex hash
		vt_hash[fj[2]].append(i)
#		i += 1
#		print(('vt_hash', vt_hash))
#		print(('fc_hash', fc_hash))
#		print()
		
	i = 1
	while i < cn_table_length:	
#	for i in range(1, cn_table_length):
		
		# which face am I now ?
#		j = 10 # need to compute this index
		#fj = faces[j-1]
#		print(len(cn_table))
#		print(cn_table_length)
#		print(i)
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
	
	# TODO
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
		ci[7] = -1 if not len(ck) else cn_table[cn_table[ck.pop(), 3], 3]
		

		# left corner
		# select c_j \ c_j_vertex == c_i_p_vertex
		# select c_k \ c_k_vertex == c_i_vertex
		# filter to c_j \ c_j_p == c_k
		# THEN c_i_right = c_j_n
		cjs = vt_hash[ci_p[1]]
		ck = vt_hash[ci[1]]
		cj_p = set(cn_table[cjs, 4])
		cj = cj_p.intersection(ck)
		ci[6] = -1 if not len(cj) else cn_table[cn_table[cj.pop(), 3], 3]


	for i in range(1, cn_table_length):
		
		# opposite corner
		# corner_i => next => right
		ci = cn_table[i,:]
		cn_right = cn_table[ci[3], 7]
		ci[5] = -1 if cn_right == -1 else cn_right
	
	fc_hash = None
	
	
	return (vt_hash, cn_table)


def closure(cnt, vt_hash, vt=[], ed=[], fc=[]):
	"""
####
# Get the closure for the vertices, edges and faces specified
###
# cnt:		The corner table
# vt_hash:	The vertices hash, given a vertex return the list of its corners
# vt:		List of vertices
# ed:		List of edges. Every edge is specified by a tuple containing its vertex, i.e., (V1, V2)
# fc:		List of faces. 
###
# 
	"""
	closure = {'vertices': set(), 'edges':set(), 'faces':set()}
	
	# vertex closure is the vertex itself
	closure['vertices'] = set(vt)
	
	# edge closure are the vertex that form the edge
	# TODO To see if should consider the edges formed from the corners
	for e in ed:
		print(e)
		closure['vertices'].add( list(e)[0] )
		closure['vertices'].add( list(e)[1] )
		closure['edges'].add( frozenset(e) )
	
	# face closure
	for f in fc:
		# get the corners for the given face f
		cns = np.where(cnt[:, 2] == f)[0]
		# if the face does not exist in the corner table continue
		if not len(cns):
			continue
		# get the vertices for the given face, column 1 in the corner table
		v = cnt[cns, 1]
		closure['vertices'].add( v[0] )
		closure['vertices'].add( v[1] )
		closure['vertices'].add( v[2] )
		# supposing the corners are sorted, then the following edges are in the correct order
		closure['edges'].add( frozenset((v[0], v[1])) )
		closure['edges'].add( frozenset((v[1], v[2])) )
		closure['edges'].add( frozenset((v[2], v[0])) )
		# add the face to its closure
		closure['faces'].add( f )
	
	return closure




def star(cnt, vt_hash, vt=[], ed=[], fc=[]):
	"""
	"""
	star = {'vertices':set(), 'edges':set(), 'faces':set()}
	
	# first break the faces and edges in its constituints
	# in order to avoid repetition, turn all structures into sets
	vt = set(vt)
	ed = set([frozenset(i) for i in ed])
	fc = set(fc)
	
	# iterate over the faces to add its edges and vertices
	for f in fc:
		# get the corners, actually the lines on the corner table,
		# that has the same face as f
		cns = np.where(cnt[:,2] == f)[0]
		# just to avoid error, check if there is any element on cns
		if len(cns) == 0:
			continue
		# add the edges on ed set
		ed.add( frozenset((cnt[cns[0], 1], cnt[cns[1], 1])) )
		ed.add( frozenset((cnt[cns[0], 1], cnt[cns[2], 1])) )
		ed.add( frozenset((cnt[cns[1], 1], cnt[cns[2], 1])) )
		# add the vertices on vt set
		vt.add( cnt[cns[0], 1] )
		vt.add( cnt[cns[1], 1] )
		vt.add( cnt[cns[2], 1] )
	
	# iterate over the edges to add its vertices
	for e in ed:
		vt.add( list(e)[0] )
		vt.add( list(e)[1] )
	
	print(('vt', vt))
	print(('ed', ed))
	print(('fc', fc))
	print()

	# TODO since I make this break-up I may not need some further steps, review
	
	# the star considering the vertex
	# we have to consider all structures, vertices, edges, and faces
	for v in vt:
		# vertex star is the vertex itself
		star['vertices'].add(v)
		
		# regarding edges
		# get the corners of this vertex
		cns = vt_hash[v]
		# get the edges with vertices leaving or arriving at v
		[star['edges'].add( frozenset((v, v_next)) ) for v_next in cnt[cnt[cns, 3], 1]]
		[star['edges'].add( frozenset((v_prev, v)) ) for v_prev in cnt[cnt[cns, 4], 1]]
		
		# regarding the faces that contain this vertex v
		# already have this info on cns, the corners that have this vertex
		[star['faces'].add( f ) for f in cnt[cns, 2]]

	print(('st_vt', star['vertices']))
	print(('st_ed', star['edges']))
	print(('st_fc', star['faces']))
	print()
	
	# the star of the edges
	for e in ed:
		
		# include the vertex at the beginning and end of the edge
		star['vertices'].add( list(e)[0] )
		star['vertices'].add( list(e)[1] )
		
		# considering that the edges are all with the same orientation, and composed by its vertices
		# since in the beginning of the function already converted ed to a set of frozenset
		# I can just add e here
		star['edges'].add( e )
		
		# get all corners with the vertex of the edge
		cns = [vt_hash[list(e)[0]], vt_hash[1]]
		# get the corner from cns[0] which the next corner is on cns[1]
		c_next = set(cnt[cns[0], 3])
		cn = c_next.intersection(cns[1])
		None if len(cn) == 0 else star['faces'].add( cnt[cn.pop(), 2] )
		# now invert cns[0] for cns[1] to get the other face
		c_next = set(cnt[cns[1], 3])
		cn = c_next.intersection(cns[0])
		None if len(cn) == 0 else star['faces'].add( cnt[cn.pop(), 2] )
		
	print(('st_vt', star['vertices']))
	print(('st_ed', star['edges']))
	print(('st_fc', star['faces']))
	print()
	
	# the star of the faces
	for f in fc:
		star['faces'].add(f)
		
		# TODO maybe I should split the vertex, edges, and faces code for the star in order to reuse here
		# not necessary
		
		# since I dont have a index structure for the faces I need to seek in the corner table
		cns = np.where(cnt[:,2] == f)[0]
		vs = cnt[cns, 1]
		
		# add the vertex of the corners
		[star['vertices'].add( v ) for v in vs]
		
		# add the edges of the face f
		star['edges'].add( frozenset((vs[0], cnt[cnt[cns[0], 3], 1] )) )
		star['edges'].add( frozenset((cnt[cnt[cns[0], 4], 1], vs[0])) )
		star['edges'].add( frozenset((cnt[cnt[cns[0], 3], 1], cnt[cnt[cns[0], 4], 1])) )
		
		# now get the surrounding faces
		# get the faces for every opposite corner (column 5 from the corner table)
		[star['faces'].add( cnt[c[5], 2] ) for c in cnt[cns,:] if c[5] != -1]
	
	
	return star

# Dont need, python already have a builtin set difference method 
def complement():
	pass


def link(cnt, vt_hash, vt=[], ed=[], fc=[]):
	link = {'vertices':set(), 'edges':set(), 'faces':set()}
	
	# TODO test if this approach does not have a complexity too high
	cls = closure(cnt=cnt, vt_hash=vt_hash, vt=vt, ed=ed, fc=fc)
	str = star(cnt=cnt, vt_hash=vt_hash, vt=vt, ed=ed, fc=fc)
	cls_str = closure(cnt=cnt, vt_hash=vt_hash, vt=str['vertices'], ed=str['edges'], fc=str['faces'])
	str_cls = star(cnt=cnt, vt_hash=vt_hash, vt=cls['vertices'], ed=cls['edges'], fc=cls['faces'])

	print(('vt', vt))
	print(('ed', ed))
	print(('fc', fc))
	print()
	print(('cls', cls))
	print(('str', str))
	print(('cls_str', cls_str))
	print(('str_cls', str_cls))

	# perform the set difference between the closure of the and the star of the closure
	# meaning the operation close(star(GAMMA)) \ star(close(GAMMA))
	# where GAMMA is the query object
	"""
	for l in link:
#		link[l] = str_cls[l].difference( cls_str[l] )
		link[l] = cls_str[l].difference( str_cls[l] )
	"""
	l = 'vertices'
	link[l] = cls_str[l].difference( str_cls[l] )
	l = 'edges'
	link[l] = cls_str[l].difference( str_cls[l] )
	l = 'faces'
	link[l] = cls_str[l].difference( str_cls[l] )
#	"""
	return link



def ring_1(cnt, vt_hash, vt=[], ed=[], fc=[]):
	"""
####
# Get the 1-ring
###
# 
###
# Return the vertices of the link
# since the 1-ring is the set of vertices in the link
	"""
	return link(cnt=cnt, vt_hash=vt_hash, vt=vt, ed=ed, fc=fc)['vertices']



def intersect_face(fc, cnt, vt_hash):
	# given a triangle, return the triangles that intersect it, vertex and edges included
	pass


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
	
	# TODO modify to read the vtk format exported from paraview

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
