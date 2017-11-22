
import sys, imp
import numpy as np
from numpy import linspace, hstack, vstack, array, ndarray, ones, average, cos, sin, pi, sqrt
from numpy.linalg import det
from numpy.random import choice as sample
from matplotlib import pyplot as plt

import corner_table as cnt


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
	outer_tr = outer_triangle(pts)
	# add the outer triangle to the corner table
	cn_table.add_triangle(outer_tr)
	# get a random permutation of the points
	pts_sample = sample(range(len(pts)), size=len(pts), replace=False, p=None)
	# iterate over all points
	for p_i in pts_sample:
		p = pts[p_i]
		# get the triangle containing the point p
		tr_p = cn_table.find_triangle(p)
		v0, v1, v2 = tr_p['vertices']
		
		# get the triangles sharing edges
		tr_shar_ed = cn_table.triangles_share_edge(eds=((v0,v1), (v1,v2), (v2,v0)))
		
		# check if the point lies on an edge, just see if there is a zero within the baricentric coords
		
		# determine the triangles to be added, if 3 or 4, and determine
		# which triangles should be removed, if 1 or 2
		
		# remove the triangles
		
		
		# add the triangles
		
		# legalize edges
		
	pass


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

