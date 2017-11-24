
import sys, imp
import numpy as np
from numpy import linspace, hstack, vstack, array, ndarray, ones, average, cos, sin, pi, sqrt
from numpy.linalg import det
from numpy.random import choice as sample
from matplotlib import pyplot as plt

import corner_table as cnt
imp.reload(cnt)

def delaunay_triangulation(pts):
	"""
#########################################
# Perform the Delaunay triangulation a set of points
#########################################
# points:	List. Each element is a n-dimensional point
#########################################
# Return a structure with the delaunay triangulation.

	"""
	# initialize the corner table
	cn_table = cnt.CornerTable()
	# compute the outer triangle
	outer_tr = [tuple(i) for i in outer_triangle(pts)]
	print(outer_tr)
	# add the outer triangle to the corner table
	cn_table.add_triangle(outer_tr)
	# get a random permutation of the points
	pts_sample = sample(range(len(pts)), size=len(pts), replace=False, p=None)
	pts_sample = range(len(pts))
	# iterate over all points
	for p_i in pts_sample:
		# p holds the physical position of the i-th point
		p = pts[p_i]
		print()
		print(('p_i', p_i, 'p', p))
		# get the triangle containing the point p
		tr_p = cn_table.find_triangle(p)
		v0, v1, v2 = tr_p['vertices']
		
		# get the triangles sharing edges
		tr_shar_ed = cn_table.triangles_share_edge(eds=((v0,v1), (v1,v2), (v2,v0)))
		
		# triangles to be added, in the case the point does not lie on some edge
		add_faces = [
						[tr_p['physical'][0], p, tr_p['physical'][2]],  
						[tr_p['physical'][0], tr_p['physical'][1], p],  
						[p, tr_p['physical'][1], tr_p['physical'][2]]
					]
		
		# check if the point lies on an edge, just see if there is a zero within the baricentric coords
		rem_faces = [ tr_p['face'] ]
		if tr_p['bari'][0] * tr_p['bari'][1] * tr_p['bari'][2] == 0:
			# determine the triangles to be added, if 3 or 4, and determine
			# which triangles should be removed, if 1 or 2
			
			# remove the triangle with zero area
			add_faces.pop( 2 if tr_p['bari'][0] == 0 else 0 if tr_p['bari'][1] == 0 else 1 )
			
			# result in the opposing vertex of the vertex with baricentric coordinate zero
			opposing_vertex = set(tr_share_ed['physical'][1])
			[opposing_vertex.discard(v) for v in tr_p['physical']]
			opposing_vertex = tuple(opposing_vertex.pop())
			
			# add the 2 new triangles to be added
			[add_faces.append([v, p, opposing_vertex]) 
					for v in set(tr_share_ed['physical'][1]).intersection(tr_p['physical'])]
			
			# define the faces to remove based on the zero of the baricentric coordinate
			# if the first coordinate if zero, then remove the second triangle on the list tr_share_ed
			# if the second coordinate if zero, then remove the third triangle on the list tr_share_ed
			# if the third coordinate if zero, then remove the first triangle on the list tr_share_ed
			rem_faces.append( tr_share_ed['faces'][ 
								1 if tr_p['bari'][0] == 0 else 1 if tr_p['bari'][1] == 0 else 2 
							])
#			tr_share_ed['faces'].pop( 
#								0 if tr_p['bari'][0] == 0 else 1 if tr_p['bari'][1] == 0 else 2
#								)
		
		# remove the triangles
		[cn_table.remove_triangle(f) for f in rem_faces]
		
		# add the triangles
		print(('ADD_FACES', add_faces))
		added_faces = [cn_table.add_triangle(f) for f in add_faces]
		
		# legalize edges
		print(('CN', cn_table._cn_table))
		check_faces = []
		for f in added_faces:
			vs = f['vertices']
			check_faces.append( cn_table.triangles_share_edge(
										eds=((vs[0],vs[1]), (vs[1],vs[2]), (vs[2],vs[0])) )['virtual'] 
								)
		legalize(check_faces)
		
	return cn_table


def legalize(face0, face1):
	print('\nLEGALIZE')
	print(face0)
	print(face1)
	print()


def orientation(points):
	"""
#########################################
# Determine the orientation of the array of points
#########################################
# points:	List. Each element is a n-dimensional point
#########################################
# Return True (counterclockwise) or False (clockwise or colinear/coplanar)
	"""
	mat = ndarray(buffer=ones(len(points)**2), shape=(len(points), len(points)))
	mat[0:len(points), 0:len(points[0])] = points
	
#	return mat
#	return det(mat)
	return det(mat) > 0



def inCircle(points):
	"""
#########################################
# Determine if the 4th point of the array of points lie inside the circle defined for the first 3 points
#########################################
# points:	List. Each element is a 2-dimensional point. The points should the in a counterclockwise orientation
#########################################
# Return True or False. The points should the in a counterclockwise orientation

# TODO generalize to more dimensions
	"""
	mat = ndarray(buffer=ones(16), shape=(4,4))
	pts = array(points)
	mat[0:4, 0:2] = pts[0:4, 0:2]
	mat[0:4, 2] = [sum([i**2 for i in p]) for p in pts]
	
	return det(mat) >= 0



def outer_triangle(pts):
	"""
#########################################
# Determine the points of a outer triangle containing all the points
#########################################
# pts:	List. Each element is a n-dimensional point
#########################################
# Return a list of points

# TODO generalize for more dimensions
	"""
	data = array(pts)
	# get the center of the data, the mean over each coordinate
	center = data.mean(axis=0)
	data = data - center
	radius = max([sqrt(p.dot(p)) for p in data])
	
	# to rotate counterclockwise
	theta = -2*pi/3
	rot_mat = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	
	p0 = [center[0], 3*radius]
	p1 = rot_mat.dot(p0)
	p2 = rot_mat.dot(p1)
	
	return (array([p0, p1, p2]) + center)

