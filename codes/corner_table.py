
import sys
from numpy import linspace, hstack, vstack, array, ndarray, ones, average, cos, sin, pi, sqrt, where, append, delete
from numpy.random import choice as sample
from numpy.linalg import det, solve
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
	_coord_hash = {'phys':{}, 'vir':[]}
	
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
	


	def test_delaunay(self):
		"""

# Usage example:

rd_pts
grd_truth


imp.reload(cnt); pp = cnt.CornerTable()
[pp.add_triangle([tuple(i) for i in rd_pts[t]]) for t in grd_truth.simplices]
pp.plot()
pp.test_delaunay()

		"""
		total_faces = 0
		oriented_faces = 0
		delaunay_faces = 0
		for f in self._fc_hash:
			if len(f) == 0:
				continue
			total_faces += 1
			
			c0,c1,c2 = f
			c0_o,c1_o,c2_o = self._cn_table[[c0,c1,c2], 5]
			
			v0,v1,v2 = self._cn_table[[c0,c1,c2], 1]
			v0_o,v1_o,v2_o = self._cn_table[[c0_o,c1_o,c2_o], 1]
			
			p_v0,p_v1,p_v2 = array(self._coord_hash['vir'])[[v0,v1,v2]]
			p_v0_o,p_v1_o,p_v2_o = array(self._coord_hash['vir'])[[v0_o,v1_o,v2_o]]
			
			# compute the number of rightly oriented faces
			oriented_faces += 1 if self.orientation([p_v0, p_v1, p_v2]) else 0
			
			# 
			delaunay_faces += 1 if self.inCircle([p_v0, p_v1, p_v0_o, p_v2]) and self.inCircle([p_v1, p_v2, p_v1_o, p_v0]) and self.inCircle([p_v2, p_v0, p_v2_o, p_v1]) else 0
		
		return (total_faces, oriented_faces, delaunay_faces)


	
	def legalize(self, point, face, plot=False):
		"""
###
# Verify if the edge need to be legalized and do
###
# point:	Tuple. The physical coordinates of the added point
# face:		Integer. The number of the added triangle
###
# Modify the corner table

# Usage example:


import corner_table as cnt


# triangles from the slides defining the corner table data structure
tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]
tr5 = [(1,0), (5,0), (3,1,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)
c_table.add_triangle(tr2)
c_table.add_triangle(tr3)
c_table.add_triangle(tr4)
c_table.add_triangle(tr5)

#c_table.plot(show=False, subplot={'nrows':1, 'ncols':2, 'num':1})
c_table.legalize(point=(0,1), face=0, plot=True)
#c_table.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2})
		"""
		# get the index of the added vertex
		vi = self._coord_hash['phys'][point]
		print(('point', point, 'face', face, 'vi', vi))
		
		# get the vertices of the given face
		corners = self._fc_hash[face]
#		if len(corners) == 0:
#			return True
		v0, v1, v2 = self._cn_table[corners, 1]
#		opposite_edge = set([v0,v1,v2])
#		opposite_edge.discard(vi)
#		opposite_edge = list(opposite_edge)
#		print(('vi', vi, 'v0,v1,v2', v0, v1, v2, 'opposite_edge', opposite_edge))
		print(('vi', vi, 'v0,v1,v2', v0, v1, v2, ))
		

		# maybe dont need this
		# get the corner of point
		if not(vi == v0 or vi == v1 or vi == v2):
			return True
		ci = corners[ [v0,v1,v2].index(vi) ]
		# get the opposite vertex of ci		
		ci_opp = self._cn_table[ci, 5]
		# if there is no opposite vertex the bondaury was reached
		if ci_opp == -1:
			return True
		# get the vertex index for the oppositve corner of ci
		opposite_vertex = self._cn_table[ci_opp, 1]

		P_v0, P_v1, P_v2 = array(self._coord_hash['vir'])[[v0,v1,v2]]
#		return True
		# check if the 4 points are in a legal arrengement
		oriented_pts = array([P_v2, P_v0, P_v1, self._coord_hash['vir'][opposite_vertex] ])
		
		print('\n\t\t\tOrientation %s'%(self.orientation(oriented_pts)))
		print(('oriented_pts', oriented_pts))
#		plt.plot(oriented_pts[:,0], oriented_pts[:,1]); [plt.text(p[0], p[1], str(i)) for i,p in enumerate(oriented_pts)]; plt.show(); plt.close('all')
		print()
		if self.inCircle(oriented_pts):
			# perform the edge flip
#			faces = self.triangles_share_edge(eds=[opposite_edge])['faces']
			faces = (self._cn_table[ci, 2], self._cn_table[ci_opp, 2])
			print(faces)
#			if len(faces) == 0:
#				return True
#			faces = faces[0]
			# before perform the slip get the MAYBE DONT NEED
			self.plot(show=False, subplot={'nrows':1, 'ncols':2, 'num':1}) if plot else 0
			flipped_fcs = self.flip_triangles(faces[0], faces[1])
			
			print(('flipped_fcs', flipped_fcs))
			self.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2}) if plot else 0
#			input('waiting')
			print()
			if flipped_fcs[0]['face'] > 100:
				return True
			
			# call legalize for the 2 other edges
			self.legalize(point, flipped_fcs[0]['face'])
			self.legalize(point, flipped_fcs[1]['face'])



	def triangles_share_edge(self, eds=[], cns=[]):
		"""
###
# Get the triangles that share edges
###
# eds:	List. The list of edges
# cns:	List. The list of corners
###
# Return a list of triangles
# It iterate over the vertices of each triangle in the order they were created
		"""
		trs = {'faces':[], 'physical':[], 'virtual':[]}
		corners = cns
		
		# convert the edges to corners
		if len(eds) > 0:
			corners = []
			for e in eds:
				c0 = set(self._cn_table[self._vt_hash[e[1]], 4]).intersection( self._vt_hash[e[0]] )
				# check if the edge exists
				if len(c0) > 0:
					c0 = c0.pop()
					c1 = self._cn_table[c0, 3]
					corners.append( (c0,c1) )
		
		for c in corners:
			# get the face of c0
			f0 = self._cn_table[c[0], 2]
			# get the face of the right corner of c0, that is equvalent to the opposite
			# of the previous corner of c0
			f1 = self._cn_table[ self._cn_table[c[0], 7] , 2]
			trs['faces'].append((f0,f1))
			
#			print(('f0', f0, 'f1', f1))
#			print(('self._fc_hash[f]', self._fc_hash[f]))
			
			trs['physical'].append([
								[tuple(self._coord_hash['vir'][v]) 
										for v in self._cn_table[self._fc_hash[f], 1] 
									] 
								for f in [f0, f1]])
			trs['virtual'].append([ tuple( self._cn_table[self._fc_hash[f], 1] )
								for f in [f0, f1]])
		return trs
		
		# THIS PART IS NOT USED ANYMORE
		"""
			# get the opposite corner (colunm 5) of the c0 (_fc_hash[f][0])
			c0 = self._cn_table[c[0], 5]
			# get the opposite corners of the next and previous corners
			cn, cp = self._cn_table[self._cn_table[c0, 3], 5], self._cn_table[self._cn_table[c0, 4], 5]
			# get the faces of the opposite corners c0, cn and cp
			fc0, fc1, fc2 = self._cn_table[[c0, cn, cp], 2]
			trs.append((-1 if c0 == -1 else fc0, -1 if cn == -1 else fc1, -1 if cp == -1 else fc2))
		return trs
		"""
	

	def find_triangle(self, point):
		"""
###
# Given a point return the triangle containing the given point
###
# point:	Tuple. The physical coordinate of the point
###
# return the number of the face

# Usage example:

import corner_table as cnt


# triangles from the slides defining the corner table data structure
tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)
c_table.add_triangle(tr2)
c_table.add_triangle(tr3)
c_table.add_triangle(tr4)

# should return the first triangle
c_table.find_triangle((1,1))

# should return the second triangle
c_table.find_triangle((2,1))

# should return the third triangle
c_table.find_triangle((3,1.5))

# should return the forth triangle
c_table.find_triangle((4,1))


		"""
#		self._vt_hash
#		self._fc_hash
#		self._cn_table
		tr_ret = -1
		# performs the search over all triangles in an random order
		# TODO it is possible to improve this searching looking for the big triangles first
		# 	maybe change the probability of the sampling by the area of the triangle
		tr_ids = sample(range(len(self._fc_hash)), size=len(self._fc_hash), replace=False, p=None)
		tr_tested = []
		print(('FC_HASH', self._fc_hash))
		print(('TR_IDS', tr_ids))
#		for tr_i in :
		while len(tr_ids) > 0:
			tr_i = tr_ids[0]
			tr_ids = tr_ids[1:]
			print(('TR_I', tr_i))
			# if the current triangle was already tested continue
			if tr_i in tr_tested or len(self._fc_hash[tr_i]) == 0:
				continue
			tr_tested.append(tr_i)
			# get the vertices of the given triangle
#			v0, v1, v2 = self._cn_table[self._fc_hash[tr_i], 1]
			c0 = self._fc_hash[tr_i][0]
			# get the physical coordinate of the 3 vertices given their "virtual" indices
			v0, v1, v2 = self._cn_table[[c0, self._cn_table[c0, 3], self._cn_table[c0, 4]], 1]
			P_v0 = tuple(self._coord_hash['vir'][v0])
			P_v1 = tuple(self._coord_hash['vir'][v1])
			P_v2 = tuple(self._coord_hash['vir'][v2])
			# perform the incircle test
			# if false continue
			# ATTENTION garantee the order of the vertices are correct
			if not self.inCircle([P_v0, P_v1, P_v2, point]):
				continue
			# get the triangles sharing edges
			#	maybe I should get all triangles with that share vertices, the clousure
			tr_cls = self.closure(fc=[tr_i])
			# for each selected triangle, there are 4 of them, find the baricentric coordinates
			# of the query point
			for tr in tr_cls['faces']:
				c0 = self._fc_hash[tr][0]
				v0, v1, v2 = self._cn_table[[c0, self._cn_table[c0, 3], self._cn_table[c0, 4]], 1]
				P_v0, P_v1, P_v2 = array(self._coord_hash['vir'])[ [v0,v1,v2] ]
				bari_c = self.bari_coord([P_v0, P_v1, P_v2], point)
				# if all baricentric coordinates are positive it mean the point is inside this triangle
				if bari_c[0] * bari_c[1] * bari_c[2] >= 0:
					
					return {'face':tr, 'vertices':[v0,v1,v2], 'bari':bari_c, 'physical':[P_v0,P_v1,P_v2]}
					break
			
			# dont need this, since if the incircle test returns ok then the point will be find in this iteration
#			[tr_tested.append(i) for i in tr_cls[]]
		return {'face':-1, 'vertices':(), 'bari':()}
		return tr_ret

	
	@staticmethod
	def bari_coord(points, query_pt):
		"""
###
# Determine the baricentric coordinates of the query point
###
# points:	List. Each element is a 2-dimensional point. The points should the in a counterclockwise orientation
# query_pt:	Tuple. The query point that will have its baricentric coordinate computed
###
# Return a tuple with the baricentric coordinate

		"""
		A = [[1, 1, 1], [points[0][0], points[1][0], points[2][0], ], [points[0][1], points[1][1], points[2][1], ]]
		return solve(A, (1, query_pt[0], query_pt[1]))


	@staticmethod
	def inCircle(points):
		"""
###
# Determine if the 4th point of the array of points lie inside or in the circle defined for the first 3 points
###
# points:	List. Each element is a 2-dimensional point. The points should the in a counterclockwise orientation
###
# Return True or False. The points should the in a counterclockwise orientation

# TODO generalize to more dimensions
		"""
		mat = ndarray(buffer=ones(16), shape=(4,4))
		pts = array(points)
		mat[0:4, 0:2] = pts[0:4, 0:2]
		mat[0:4, 2] = [sum([i**2 for i in p]) for p in pts]
		
		return det(mat) > 0



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
#		print(('VERTICE', vertice))
#		print(('_coord_hash', self._coord_hash))
		if not tuple(vertice) in self._coord_hash['phys'].keys():
			self._coord_hash['phys'][vertice] = len(self._coord_hash['phys'])
			self._coord_hash['vir'].append(vertice)
			self._vt_hash.append( [] )
		return self._coord_hash['phys'][tuple(vertice)]





	def flip_triangles(self, face0, face1):
		"""
###
# Perform the flip of two adjacent triangles
###
# face0:	Number. The index of the face
# face1:	Number. The index of the face
###
# Modify the current corner table
# This method removes the edges between the two triangles and
# add a new one between the opposing corners, performing the edge flip

# Usage example:

import corner_table as cnt


# triangles from the slides defining the corner table data structure
tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]
tr5 = [(1,0), (5,0), (3,1,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)
c_table.add_triangle(tr2)
c_table.add_triangle(tr3)
c_table.add_triangle(tr4)
c_table.add_triangle(tr5)

c_table.plot(show=False, subplot={'nrows':1, 'ncols':2, 'num':1})
c_table.flip_triangles(1, 2)
c_table.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2})
		"""
		
		# TODO check if it is possible to split the triangles
		
		# TODO having problems when remove a triangle and a vertex is left hanging,
		# this happens for triangles on the boundary

		
		# the easiest way to implement this is to remove both faces and add the new ones
		
		# first get the vertices
		v0, v1, v2 = self._cn_table[self._fc_hash[face0], 1]
		v3, v4, v5 = self._cn_table[self._fc_hash[face1], 1]
		# get the vertices repeted between face0 and face1
		v_rep_01 = list(set((v0,v1,v2)).intersection((v3,v4,v5)))
		
		# get the opposing vertices for faces 0 and 1
		v_opp_0 = set((v3,v4,v5))
		v_opp_1 = set((v0,v1,v2))
		[v_opp_0.discard(i) for i in (v0,v1,v2)]
		[v_opp_1.discard(i) for i in (v3,v4,v5)]
		v_opp_0 = v_opp_0.pop()
		v_opp_1 = v_opp_1.pop()
		
		print(('v0,v1,v2', v0,v1,v2))
		print(('v3,v4,v5', v3,v4,v5))
		print(('v_rep_01', v_rep_01))
		print(('v_opp_0', v_opp_0))
		print(('v_opp_1', v_opp_1))
		
		# build the new triangles
		# I have to specify the physical position of the vertices, not the edges
		# as I was trying before
		tr0 = [self._coord_hash['vir'][v_opp_1], self._coord_hash['vir'][v_opp_0], self._coord_hash['vir'][v_rep_01[0]]]
		tr1 = [self._coord_hash['vir'][v_opp_0], self._coord_hash['vir'][v_opp_1], self._coord_hash['vir'][v_rep_01[1]]]
		
		print(('tr0', tr0))
		print(('tr1', tr1))
	
		# remove face0 and face1 and add tr0 and tr1
		self.remove_triangle(face0)
		self.remove_triangle(face1)
		face0 = self.add_triangle(tr0)
		face1 = self.add_triangle(tr1)

		return (face0, face1)



	def remove_triangle(self, face):
		"""
###
# Remove a triangle from the Corner Table
###
# face:	Number. The index of the face
###
# Modify the current corner table

# Usage example:

import corner_table as cnt


# triangles from the slides defining the corner table data structure
tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)
c_table.add_triangle(tr2)
c_table.add_triangle(tr3)
c_table.add_triangle(tr4)

c_table.remove_triangle(2)
		"""
		# get the corners of the given triangle
		c0, c1, c2 = self._fc_hash[face]
		# get the vertices of these corners
		v0, v1, v2 = self._cn_table[[c0, c1, c2], 1]
		
#		print(('c0,c1,c2', c0, c1, c2))
#		print(('v0,v1,v2', v0, v1, v2))
		
		# to remove this face just need to erase the entries regarding these corners
		# from the hashs (fc_hash and vt_hash) and from the corner table
		# meaning to set -1 to the first column of each line, c0, c1, c2
		self._vt_hash[v0].remove(c0)
		self._vt_hash[v1].remove(c1)
		self._vt_hash[v2].remove(c2)
		
		self._fc_hash[face] = []
		
#		print(('self._vt_hash', self._vt_hash[v0], self._vt_hash[v1], self._vt_hash[v2]))
#		print(('self._fc_hash', self._fc_hash[face]))
		
		# fix the surrounding faces
		print(('AA', self.triangles_share_edge(cns=((c0,c1), (c1,c2), (c2,c0))) ))
		surrounding_faces = list(set([t[1] for t in self.triangles_share_edge(cns=((c0,c1), (c1,c2), (c2,c0)))['faces']]))
#		surrounding_faces = [f for f in set([t[1] for t in self.triangles_share_edge(cns=((c0,c1), (c1,c2), (c2,c0)))['faces']]) if f[0] != f[1]]
		surrounding_corners = array(self._fc_hash)[surrounding_faces]
		self._cn_table[[c0,c1,c2], 0] = -1
#		print(('surrounding_faces', surrounding_faces))
#		print(('surrounding_corners', hstack(surrounding_corners)))
#		print(('CN[c0,c1,c2]', self._cn_table[[c0,c1,c2],]))
		self.fix_opposite_left_right(self._cn_table, self._vt_hash, ids=hstack(surrounding_corners))


	def add_triangle(self, vertices):
		"""
###
# Add a triangle to the Corner Table
###
# vertices:		List. The list of vertices, the physical coordinate of each vertice
###
# Return the index of the added triangle and its vertices indices
# Modify the current corner table

# Usage example:

import corner_table as cnt


# triangles from the slides defining the corner table data structure
tr1 = [(0,1), (1,0), (2,2,),]
tr2 = [(2,2), (1,0), (3,1,),]
tr3 = [(2,2), (3,1), (4,2,),]
tr4 = [(3,1), (5,0), (4,2,),]

imp.reload(cnt)
c_table = cnt.CornerTable()
c_table.add_triangle(tr1)
c_table.add_triangle(tr2)
c_table.add_triangle(tr3)
c_table.add_triangle(tr4)

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
#		print(self._cn_table)
#		print(cn_triangle)
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
#		print(('fix_ids', fix_ids))
#		print(self._cn_table)
#		print(self._vt_hash)
#		print(self._cn_table[fix_ids])
#		print(array(self._vt_hash)[fix_ids])
		self.fix_opposite_left_right(self._cn_table, self._vt_hash, fix_ids)
		
#		return {'face': len(self._fc_hash), 'vertices': (v0,v1,v2)}
		return {'face': fc_id, 'vertices': (v0,v1,v2)}
		

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
		ids = range(len(cn_table)) if len(ids) == 0 else [int(i) for i in ids]
		# TODO
		# JUST GIVE AN EXTRA NEXT ON THESE 3 THAT SOLVE ACCORDING TO THE EXAMPLE	

		# then compute the oposite, left and right corners
#		for i in range(1, cn_table_length):
		print(('FIX_ ids', ids))
		for i in ids:
			
			ci = cn_table[i,:]
			ci_n = cn_table[ci[3],:]
			ci_p = cn_table[ci[4],:]
			"""
			print(('i', i))
			print(('ci', ci))
			print(('ci_n', ci_n))
			print(('ci_p', ci_p))
			"""
			# right corner
			# select c_j \ c_j_vertex == c_i_n_vertex
			# select c_k \ c_k_vertex == c_i_vertex
			# filter to c_k \ c_k_p == c_j
			# THEN c_i_right = c_k_n
#			"""
			cj = vt_hash[ci_n[1]]
			cks = vt_hash[ci[1]]
#			print(('cj',cj))
#			print(('cn_table',cn_table))
#			print(('cks',cks))
			ck_p = set(cn_table[cks, 4])
#			print(('ck_p',ck_p))
			ck = ck_p.intersection(cj)
#			print(('ck',ck))
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


	def closure(self, vt=[], ed=[], fc=[]):
		"""
	####
	# Get the closure for the vertices, edges and faces specified
	###
	# vt:		List of vertices
	# ed:		List of edges. Every edge is specified by a tuple containing its vertex, i.e., (V1, V2)
	# fc:		List of faces. 
	###
	# 
		"""
		cnt, vt_hash = self._cn_table, self._vt_hash
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




	def star(self, vt=[], ed=[], fc=[]):
		"""
		"""
		cnt, vt_hash = self._cn_table, self._vt_hash
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



	def link(self, vt=[], ed=[], fc=[]):
		cnt, vt_hash = self._cn_table, self._vt_hash
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



	def ring_1(self, vt=[], ed=[], fc=[]):
		"""
	####
	# Get the 1-ring
	###
	# 
	###
	# Return the vertices of the link
	# since the 1-ring is the set of vertices in the link
		"""
		cnt, vt_hash = self._cn_table, self._vt_hash
		return self.link(vt=vt, ed=ed, fc=fc)['vertices']



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
		
	def plot(self, show=True, subplot=None):
		import matplotlib.pyplot as plt
		edges = []
		for f in self._fc_hash:
			if len(f) == 0:
				continue
			# get the vertices indexes for every face, the ones not excluded
#			print(('f', f))
#			print(('self._cn_table[f, 1]', self._cn_table[f, 1]))
			v0,v1,v2 = self._cn_table[f, 1]
			# get the physical position of the vertices
			p0,p1,p2 = array(self._coord_hash['vir'])[ self._cn_table[f, 1] ]
#			print((' p0', p0, ' p1', p1, ' p2', p2 ))
#			edges.append(((p0,p1), (p1,p2), (p2,p0)))
			edges.append((p0,p1,p2,p0))
		edges = array(edges)
#		print(('edges', edges))
		if subplot:
			plt.subplot(subplot['nrows'], subplot['ncols'], subplot['num'])
		[plt.plot(e[:,0], e[:,1]) for e in edges]
		if show:
			plt.show()
			plt.close('all')
		return edges
		
