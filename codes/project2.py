
import sys, imp
import numpy as np
from numpy import linspace, hstack, vstack, array, ndarray, ones, average, cos, sin, pi, sqrt, where
from numpy.linalg import det
from numpy.random import choice as sample
from matplotlib import pyplot as plt

import corner_table as cnt
imp.reload(cnt)

def delaunay_triangulation(pts, plot=False, legalize_plot=False, legalize=True, remove_outer=False):
	"""
#########################################
# Perform the Delaunay triangulation a set of points
#########################################
# points:	List. Each element is a n-dimensional point
#########################################
# Return a structure with the delaunay triangulation.

# Usage example:

# simple example
import corner_table as cnt
import project2 as pjt
import imp

pts = [(0,1), (3,1), (5,0), (2,2), (4,2), (1,0)]

pts2 = [(0,1), (3,1), (5,0), (2,2), (4,2), (1,0), (3, 1.9)]

outer_tr = [(5.0, 9.0777472107017552), (8.2455342898166109, -5.2039371148119731), (-5.7455342898166091, -0.87381009588978342)]

imp.reload(pjt); dd = pjt.delaunay_triangulation(pts2)


# example insert point into an edge
import corner_table as cnt
import project2 as pjt
import imp

pts = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (1,2), (1,3),]
pts = [(1,3), (1,1), (1,2), (0,0), (1,0), (2,0), (0,1), (2,1),]
pts = [(1,3), (1,1), (1,2)]

outer_tr = [(5.0, 9.0777472107017552), (8.2455342898166109, -5.2039371148119731), (-5.7455342898166091, -0.87381009588978342)]

pts2 = vstack((outer_tr, pts))

imp.reload(pjt); dd = pjt.delaunay_triangulation(pts, legalize_plot=True)

grd_truth = Delaunay(points=pts2, furthest_site=False, incremental=True)
imp.reload(pjt); my_delaunay = pjt.delaunay_triangulation(pts)

plt.subplot(1,2,1)
plt.triplot(pts2[:,0], pts2[:,1], grd_truth.simplices.copy())
plt.suptitle('Ground truth vs. My triangulation')
my_delaunay.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2})

plt.close('all')


# example with random points
import imp
import project2 as pjt
import corner_table as cnt
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from numpy.random import uniform
from numpy import array, matrix, vstack, ndarray

imp.reload(cnt);
imp.reload(pjt)

# generate the random points with the outer triangle englobing them
low, high, size = 0, 50, 50
rd_pts = ndarray(buffer=uniform(low=low, high=high, size=2*size), dtype=float, shape=(size, 2))
outer_pts = pjt.outer_triangle(rd_pts)
rd_pts = vstack((outer_pts, rd_pts))


grd_truth = Delaunay(points=rd_pts, furthest_site=False, incremental=True)
imp.reload(pjt); my_delaunay = pjt.delaunay_triangulation([tuple(i) for i in rd_pts[3:]])
my_delaunay._clean_table()

plt.subplot(1,2,1)
plt.triplot(rd_pts[:,0], rd_pts[:,1], grd_truth.simplices.copy())
edges = my_delaunay.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2})

plt.close('all')

grd_table = cnt.CornerTable()
a=[grd_table.add_triangle([tuple(i) for i in rd_pts[t]]) for t in grd_truth.simplices]
my_delaunay.test_delaunay()
grd_table.test_delaunay()

	"""
	# initialize the corner table
	cn_table = cnt.CornerTable()
	# compute the outer triangle
	outer_tr = [tuple(i) for i in outer_triangle(pts)]
	# add the outer triangle to the corner table
	cn_table.add_triangle(outer_tr)
	# get a random permutation of the points
	pts_sample = sample(range(len(pts)), size=len(pts), replace=False, p=None)
#	pts_sample = range(len(pts))
	cn_table.plot() if plot else 0
	# iterate over all points
	for p_i in pts_sample:
		cn_table.plot(show=False, subplot={'nrows':1, 'ncols':2, 'num':1}) if plot else 0
		# p holds the physical position of the i-th point
		p = tuple(pts[p_i])
		# get the triangle containing the point p
		tr_p = cn_table.find_triangle(p)
		v0, v1, v2 = tr_p['vertices']
		
		# get the triangles sharing edges
		tr_share_ed = cn_table.triangles_share_edge(eds=((v0,v1), (v1,v2), (v2,v0)))
		
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
			
			index_bari_zero = 1 if tr_p['bari'][0] == 0 else 2 if tr_p['bari'][1] == 0 else 0
			
			# result in the opposing vertex of the vertex with baricentric coordinate zero
			opposing_vertex = set(tr_share_ed['physical'][ index_bari_zero ][1])
			[opposing_vertex.discard(tuple(v)) for v in tr_p['physical']]
			opposing_vertex = tuple(opposing_vertex.pop())
			
			# add the 2 new triangles to be added
			[add_faces.append([v, p, opposing_vertex]) 
					for v in set(tr_share_ed['physical'][ index_bari_zero ][1]).intersection([tuple(i) for i in tr_p['physical']])]
			
			# define the faces to remove based on the zero of the baricentric coordinate
			# if the first coordinate if zero, then remove the second triangle on the list tr_share_ed
			# if the second coordinate if zero, then remove the third triangle on the list tr_share_ed
			# if the third coordinate if zero, then remove the first triangle on the list tr_share_ed
			rem_faces.append( tr_share_ed['faces'][ index_bari_zero ][1])
			
		
		# remove the triangles
		[cn_table.remove_triangle(f) for f in rem_faces]
		
		# add the triangles
		added_faces = [cn_table.add_triangle(f) for f in add_faces]
		
		# legalize edges
		# legalize using the inserted point and the 3/4 triangles added
		[cn_table.legalize(point=p, face=f['face'], plot=legalize_plot) for f in added_faces] if legalize else 0
		cn_table.plot(show=True, subplot={'nrows':1, 'ncols':2, 'num':2}) if plot else 0
	
	# remove outer triangle
	# since the outer triangle is the first one, its vertices are 0,1,2
	if remove_outer:
		[cn_table.remove_triangle(t) for t in cn_table.star(vt=[0,1,2])['faces']]
	
	# clean the corner table
	cn_table._clean_table()
	
	# with the removal of the outer triangle it might lose the convex hull
	# so it is require to walk over all border vertices drawing edges between them
	
	
	
	return cn_table



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



def outer_triangle(pts, p0=False):
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
	if p0:
		p0 = data[data[:,1].argmax(), :]
		y_min = data[:,1].min() -1
	
	
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

