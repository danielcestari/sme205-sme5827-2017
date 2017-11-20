
import sys
from numpy import linspace, hstack, vstack, array, ndarray, ones, average, cos, sin, pi, sqrt, where, append
from numpy.linalg import det
import matplotlib.pyplot as plt
from grid2vtk import grid2vtk

"""
# general example

ex2 = cnt.read_vtk('example2.vtk')

ex2_corner = cnt.compute_corner_table(ex2[0], ex2[1][:, [1,2,3]])

imp.reload(cnt); cnt.plot_vtk(ex2[0], ex2[2][:,[1,2,3]], ex2_corner[1], ths=0.4, figsize=(8,6), filename='example2.png')

"""

class CornerTable:
	
	# given a vertice id, return the list of corners
	_vt_hash = []
	
	# given a physical point (x,y,z coordinates), return the vertice id
	_coord_hash = {}
	
	# given a face id, return the list of corners
	_fc_hash = []
	
	# matrix representing the corner table
	_cn_table = []
	
	def __init__(self, filename=""):
		"""
###
# Initialize the Corner Table
###
# filename:	String. The vtk filename to read from. If empty initialize an empty Corner Table
###
# return an initialized Corner Table object
		"""
		cnt = ([], [], [])
		if filename:
			data = read_vtk(filename)
			# TODO check the orientation of the read triangles
			cnt = compute_corner_table(vertices=data[0], faces=data[1])
		
		self._vt_hash, self._fc_hash, self._cn_table = cnt[0], cnt[1], array(cnt[2])

	@staticmethod
	def orientation(points):
		"""
###
# Determine the orientation of the array of points
###
# points:	List. Each element is a n-dimensional point
###
# Return True (counterclockwise) or False (clockwise or colinear/coplanar)
		"""
		mat = ndarray(buffer=ones(len(points)**2), shape=(len(points), len(points)))
		mat[0:len(points), 0:len(points[0])] = points
		return det(mat) > 0



	def _add_vertice(self, vertice):
		"""
###
# Add a vertice
###
# vertice:	Tuple. The physical location of the vertice
###
# Return the index of the added vertice, if the vertice is already in the structure
# only return its index
# Modify the current corner table
# Internal method should be used outside of the class
		"""
		if not vertice in self._coord_hash.keys():
			self._coord_hash[vertice] = len(self._coord_hash)
			self._vt_hash.append( [] )
		return self._coord_hash[vertice]



	def add_triangle(self, vertices):
		"""
###
# Add a triangle to the Corner Table
###
# vertices:		List. The list of vertices
###
# Modify the current corner table

# Usage example:

import corner_table as cnt

tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)

		"""
		# garantee to have only 3 points
		vts = vertices[:3]
		# check the orientation and reverse the order if it is not counterclockwise
		if not self.orientation(vts):
			vts.reverse()
		
		# add the z coordinate if it is missing
#		if len(vts[0]) < 3:
#			vts[:, 3] = [1,1,1]
		
		# first add the vertices to vt_hash 
		
		# get the face index add a new element to fc_hash
		fc_id = len(self._fc_hash)
		v0 = self._add_vertice(vts[0])
		v1 = self._add_vertice(vts[1])
		v2 = self._add_vertice(vts[2])
		c0 = len(self._cn_table)
		c1, c2 = c0+1, c0+2
		self._fc_hash.append( [c0, c0+1, c0+2] )
		
		# FIRST check if the vertices to be added aren't already in the structure
		self._vt_hash[v0].append( c0 )
		self._vt_hash[v1].append( c1 )
		self._vt_hash[v2].append( c2 )
		
		cn_triangle = [
						[c0, v0, fc_id, c1, c2,  -1, -1, -1],
						[c1, v1, fc_id, c2, c0,  -1, -1, -1],
						[c2, v2, fc_id, c0, c1,  -1, -1, -1],
					]
		
		# add to the structure and calls fix to garantee consistency
		print(self._cn_table)
		print(cn_triangle)
		self._cn_table = vstack([ self._cn_table, cn_triangle ]) if len(self._cn_table) != 0 else array(cn_triangle)
		
#		[self._vt_hash.append( [v[0], v[1], v[2] if len(v) == 2 else 1] ) for v in vts]
		
		# INSERT THE TRIANGLE TO THE CORNER TABLE
		# add the vertices, and the face, then calls fix_opposite_left_right 
		# passing a subset of the corner table
		# only the corners of the star of the added triangle
		#
		# Maybe the star has more elements then the needed, but it is a easy start since it is 
		# already implemented, but in the future return only the triangle that share edges
		# with the added one
		
		# just get the faces of the opposite corners to the one being added
		# then get the two next corners and you have one adjacent face
		c_ids = [self._vt_hash[v0], self._vt_hash[v1], self._vt_hash[v2]]
		fix_ids = set(hstack([
							hstack(c_ids),
							list(set(hstack([self._fc_hash[fc] for ci in c_ids for fc in self._cn_table[ci, 2]]))),
						]))
		print(('fix_ids', fix_ids))
		print(self._cn_table)
		print(self._vt_hash)
#		print(self._cn_table[fix_ids])
#		print(array(self._vt_hash)[fix_ids])
		self.fix_opposite_left_right(self._cn_table, self._vt_hash, fix_ids)
		

	@staticmethod	
	def fix_opposite_left_right(cn_table, vt_hash, ids=[]):
		"""
###
# Fix the entries opposite, left and right of the corner table
###
# cn_table:	
# bt_hash:	
###
# Modify the current corner table
		"""	
		ids = range(len(cn_table)) if len(ids) == 0 else ids
		# TODO
		# JUST GIVE AN EXTRA NEXT ON THESE 3 THAT SOLVE ACCORDING TO THE EXAMPLE	

		# then compute the oposite, left and right corners
#		for i in range(1, cn_table_length):
		for i in ids:
			
			ci = cn_table[i,:]
			ci_n = cn_table[ci[3],:]
			ci_p = cn_table[ci[4],:]
			
			print(('i', i))
			print(('ci', ci))
			print(('ci_n', ci_n))
			print(('ci_p', ci_p))
			
			# right corner
			# select c_j \ c_j_vertex == c_i_n_vertex
			# select c_k \ c_k_vertex == c_i_vertex
			# filter to c_k \ c_k_p == c_j
			# THEN c_i_right = c_k_n
#			"""
			cj = vt_hash[ci_n[1]]
			cks = vt_hash[ci[1]]
			print(('cj',cj))
			print(('cn_table',cn_table))
			print(('cks',cks))
			ck_p = set(cn_table[cks, 4])
			print(('ck_p',ck_p))
			ck = ck_p.intersection(cj)
			print(('ck',ck))
			"""
			
			n_c_v_ci_n = cn_table[vt_hash[ci_n[1]], 3]
			c_v_ci = set(vt_hash[ci[1]])
			ck = c_v_ci.intersection(n_c_v_ci_n)
			"""
			
			ci[7] = -1 if not len(ck) else cn_table[cn_table[ck.pop(), 3], 3]
#			ci[7] = -1 if not len(ck) else cn_table[ck.pop(), 3]
			

			# left corner
			# select c_j \ c_j_vertex == c_i_p_vertex
			# select c_k \ c_k_vertex == c_i_vertex
			# filter to c_j \ c_j_p == c_k
			# THEN c_i_right = c_j_n
#			"""
			cjs = vt_hash[ci_p[1]]
			ck = vt_hash[ci[1]]
			cj_p = set(cn_table[cjs, 4])
			cj = cj_p.intersection(ck)
			"""
			
			n_c_v_ci_p = cn_table[vt_hash[ci_p[1]], 3]
			c_v_ci = set(vt_hash[ci[1]])
			ck = c_v_ci.intersection(n_c_v_ci_p)
			"""
			
			ci[6] = -1 if not len(cj) else cn_table[cj.pop(), 4]
#			ci[6] = -1 if not len(ck) else cn_table[ck.pop(), 4]


#		for i in range(1, cn_table_length):
		for i in ids:
			
			# opposite corner
			# corner_i => next => right
			ci = cn_table[i,:]
			cn_right = cn_table[ci[3], 7]
			ci[5] = -1 if cn_right == -1 else cn_right




	def compute_corner_table(vertices, faces, oriented=True, cn_table=[], vt_hash=[], fc_hash=[]):
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
		if not cn_table:
			cn_table_length = 3*len(faces) +1
			cn_table = ndarray(shape=(cn_table_length, 8), dtype=int, buffer=array([-1]*cn_table_length*8))
		else:
			# ADD the number of required lines in cn_table
			cn_table = vstack([cn_table, [[-1]*len(cn_table[0])]*len(vertices) ])

		# create vertex and face hash
		# a structure used for, given a vertex retrieve the associated corners
		# vt_hash is returned as the list of corners for each vertex
#		vt_hash, fc_hash = [], []
		[vt_hash.append([]) for i in range(len(vertices)+1)]
		[fc_hash.append([]) for i in range(len(faces)+1)]

		# the columns of the corner table are:
		# | corner | vertice | triangle | next | previous | opposite | left | right |
		
		# first construct all corners with its vertices, faces, next and previous corners
	#	for i in range(1, cn_table_length+1):
		i = 1
		i = len(cn_table) if len(cn_table) else 1
		for j in range(len(faces)):
			i = j*3 +1
			
			# which face am I now ?
	#		j = 10 # need to compute this index
			fj = faces[j]
			
			ci = cn_table[i,:]
			# assign the corner number, the vertex, and the face for the corner_i
			ci[0], ci[1], ci[2] = i, fj[0], j+1
			fc_hash[j+1].append( i )
			# compute the next and previous corners for the corner_i
	#		ci[3], ci[4] = fj[1], fj[2]
			# add the corner to the vertex hash
			vt_hash[fj[0]].append(i)
			i += 1
			
			ci = cn_table[i,:]
			# assign the corner number, the vertex, and the face for the corner_i+1
			ci[0], ci[1], ci[2] = i, fj[1], j+1
			fc_hash[j+1].append( i )
			# compute the next and previous corners for the corner_i+1
	#		ci[3], ci[4] = fj[2], fj[0]
			# add the corner to the vertex hash
			vt_hash[fj[1]].append(i)
			i += 1
			
			ci = cn_table[i,:]
			# assign the corner number, the vertex, and the face for the corner_i+2
			ci[0], ci[1], ci[2] = i, fj[2], j+1
			fc_hash[j+1].append( i )
			# compute the next and previous corners for the corner_i+2
	#		ci[3], ci[4] = fj[0], fj[1]
			# add the corner to the vertex hash
			vt_hash[fj[2]].append(i)
	#		i += 1
			
		i = 1
		while i < cn_table_length:	
	#	for i in range(1, cn_table_length):
			
			# which face am I now ?
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
		
		fix_opposite_left_right(cn_table, vt_hash)
		return (vt_hash, fc_hash, cn_table)
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
		
#		fc_hash = None
		
		
		return (vt_hash, fc_hash, cn_table)


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
			closure['vertices'].add( list(e)[0] )
			closure['vertices'].add( list(e)[1] )
			closure['edges'].add( frozenset(e) )
		
		# face closure
		for f in fc:
			# get the corners for the given face f
			cns = where(cnt[:, 2] == f)[0]
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
			cns = where(cnt[:,2] == f)[0]
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
		
		# the star of the faces
		for f in fc:
			star['faces'].add(f)
			
			# TODO maybe I should split the vertex, edges, and faces code for the star in order to reuse here
			# not necessary
			
			# since I dont have a index structure for the faces I need to seek in the corner table
			cns = where(cnt[:,2] == f)[0]
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



	def link(cnt, vt_hash, vt=[], ed=[], fc=[]):
		link = {'vertices':set(), 'edges':set(), 'faces':set()}
		
		# TODO test if this approach does not have a complexity too high
		cls = closure(cnt=cnt, vt_hash=vt_hash, vt=vt, ed=ed, fc=fc)
		str = star(cnt=cnt, vt_hash=vt_hash, vt=vt, ed=ed, fc=fc)
		cls_str = closure(cnt=cnt, vt_hash=vt_hash, vt=str['vertices'], ed=str['edges'], fc=str['faces'])
		str_cls = star(cnt=cnt, vt_hash=vt_hash, vt=cls['vertices'], ed=cls['edges'], fc=cls['faces'])

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
		# then convert it to list so the array can be used
		pts = array( list( csv.reader(txt[pts_init:(pts_init + pts_size)], delimiter=' ') ), dtype=pts_type)
		
		for i,st in enumerate(txt[(pts_init+pts_size):]):
			if(st.find('CELLS') != -1):
				tmp = st.split(' ')
				fcs_init = i+1
				fcs_size = int(tmp[1])
				break
		# use the csv to read and separate the columns using space as the separator
		# then convert it to list so the array can be used
		fcs = array( list( csv.reader(txt[(pts_init + pts_size + fcs_init):(pts_init + pts_size + fcs_init + fcs_size)], delimiter=' ') ), dtype=int)
		
		return (pts, fcs)



	"""
	example
	imp.reload(cnt); cnt.plot_vtk(pp[0], pp[1][:,[1,2,3]], corner2[1])
	"""
	def plot_vtk(pts, fcs, cnt, ths=0.01, figsize=(8,6), filename='./graph.png'):
		from matplotlib import pyplot as plt
		
		plt.figure(figsize=figsize)
		
		# put the points into a array structure
		xyz = array(pts)
	#	print(xyz)
		
		edges = vstack([[(f[0],f[1]), (f[1],f[2]), (f[2],f[0])] for f in fcs])
	#	print([(e[0], e[1]) for e in edges])
		
		# print the edges
		[plt.plot((xyz[e[0]-1, 0], xyz[e[1]-1, 0]), (xyz[e[0]-1, 1], xyz[e[1]-1, 1]), 'r*-') for e in edges]
		[plt.text(v[0], v[1], 'V%d'%(i+1)) for i,v in enumerate(xyz)]

		# compute the faces center and plot the names
		fcs_center = [( average(xyz[f -1, 0]), average(xyz[f -1, 1]) ) for f in fcs]
	#	print(fcs_center)
		[plt.text(f[0], f[1], 'f%d'%(i+1)) for i,f in enumerate(fcs_center)]
		
		
		# get every corner position and pull a little bit towards the face center
		# the threshold to pull the coordinates of the corner
	#	ths = 0.1
		for i,c in enumerate(cnt[1:]):
			
	#		print(('c_i', i+1, 'V', c[1], 'x', xyz[c[1]-1, 0], 'fcs_x', fcs_center[c[2]-1][0], 'y', xyz[c[1]-1, 1], 'fcs_y', fcs_center[c[2]-1][1]))
			
			x = xyz[c[1]-1, 0]*(1-ths) + fcs_center[c[2]-1][0]*ths
	#		if (xyz[c[1]-1, 0] > fcs_center[c[2]-1][0]):
	#			x = xyz[c[1]-1, 0]*(ths) + fcs_center[c[2]-1][0]*(1-ths)
			
			y = xyz[c[1]-1, 1]*(1-ths) + fcs_center[c[2]-1][1]*ths
	#		if(xyz[c[1]-1, 1] > fcs_center[c[2]-1][1]):
	#			y = xyz[c[1]-1, 1]*(ths) + fcs_center[c[2]-1][1]*(1-ths)
	#		print(c[2])
	#		x += fcs_center[c[2]-1][0]*ths if x < fcs_center[c[2]-1][0] else -fcs_center[c[2]-1][0]*ths
	#		x += -xyz[c[1]-1, 0] + (1-ths) * xyz[c[1]-1, 0]
	#		y += fcs_center[c[2]-1][1]*ths if y < fcs_center[c[2]-1][1] else -fcs_center[c[2]-1][1]*ths
	#		y += -xyz[c[1]-1, 1] + (1-ths) * xyz[c[1]-1, 1]

	#		print(('x_n', x, 'y_n', y))
	#		print()
			
			plt.text(x, y, 'C%d'%(i+1))
			
	#		fcs_center[c[2]-1][0]*ths
	#	cnts_pos = [(xyz[c[1]-1, 0] + fcs_center[c[2]-1][0]*ths, xyz[c[1]-1, 1] + fcs_center[c[2]-1][1]*ths) for i,c in enumerate(cnt[1:])]
	#	[plt.text(c[0], c[1], 'C%d'%(i+1)) for i,c in enumerate(cnts_pos)]
		
		plt.savefig(filename)
		plt.close('all')
		plt.show()
		
		
