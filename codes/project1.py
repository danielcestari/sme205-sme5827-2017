
import sys
import numpy as np
from matplotlib import pyplot as plt


def circle(resolution, center, radius):
	"""
####
# Generate a circle following counter-clockwise orientation from the rightmost point
###
# resolution:	Integer. The number of points in the circle
# center:		Tuple. The center of the circle (x,y)
# radius		Number. The radius of the circle
###
# Return a tuple with two arrays (x,y) 
	"""
	t = np.linspace(start=0, stop=2*np.pi, num=resolution)
	return (radius*np.cos(t) + center[0], radius*np.sin(t) + center[1])




def generate_curve(resolution=100, left_border=1, domain_length=5, domain_height=4, 
					curve_params={'radius':1}, equation=circle, filename=''):
	"""
####
# Generate the profile of a give curve
###
# resolution:		Integer. The number of points in the curve
# left_border:		Number. The rightmost x point
# domain_length:	Number. The total length of the domain
# domain_height:	Number. The total height of the domain
# curve_params:		Dictionary. Other parameters used for generating the curve
# equation:			Function. The function that apply the curve function
# filename:			String. The filename from which the curve will be read from
###
# Returns a tuple with x_min, x_max, y_min, y_max, curve_points

# example:
imp.reload(pjt); vv = pjt.generate_curve(100, 2, 10, 6, {'radius':1}, pjt.circle, ''); plt.plot(vv['curve'][0], vv['curve'][1], ); plt.plot([vv['center'][0]], [vv['center'][1]], '*'); plt.xlim((vv['x_min'], vv['x_max'])); plt.ylim((vv['y_min'], vv['y_max'])); plt.show(); plt.close('all')

	"""
	
	# for simplicity I'm going to let the curve in the origin (0,0) and adjust the domain based on that
	
	# TODO check for the right measures, radius smaller than the height, etc....
	
	
	if filename:
		f = open(filename, 'rt')
		curve = [i.split(' ') for i in f.read().splitlines()]
		curve = [(float(i[0]), float(i[-1])) for i in curve]
		curve = np.array(([i[0] for i in curve], [i[1] for i in curve]))
	else:
		curve = equation(resolution, (0,0), **curve_params)
	center = ( np.average(curve[0]), np.average(curve[1]))
	cv_x_min, cv_x_max = min(curve[0]), max(curve[0])
	cv_y_min, cv_y_max = min(curve[1]), max(curve[1])
	
	
	return { 	'x_min':cv_x_min - left_border, 'x_max':(cv_x_min - left_border) + domain_length,
				'y_min':-domain_height/2 , 'y_max':domain_height/2,
				'x_min_cv':min(curve[0]), 'x_max_cv':max(curve[0]),
				'y_min_cv':min(curve[1]), 'y_max_cv':max(curve[1]),
				'center':center, 'curve':curve
			}


def heuristic_1(domain, k, resolution, threshold):
	"""
####
#
##
#
# 
# resolution:		Integer. The resolution, in number of points, of each partition. Should be 
#					half the length of the curve
##
#

	"""
	
	# the resolution of the grid/border is given by the number of points in the 
	# curve
	
	# the partitions are equally spaced in X
	# the first two partitions define how much is left
	
	# if it is not possible to have all partitions equally spaced,
	# at least make the partition after the curve with the same principle
	# and then for the rest adjust its size to fit all k partitions
	
	# ATTENTION make the right border of the previous partition equal the left border
	# of the next partition
	
	# start by the division of the curve into two, then continue the partition
	
	borders = [[]] * (k+1)
	# add the leftmost point for the iteration to work later
	x_divs = [domain['x_min']]
	
	# the 3 first divisions are different, because the first one I have to check the distance
	# between the first division to the curve if it respects the threshold
	# then the second is fixed in the center of the curve
	# and the third I choose to be symmetric to the second one respect to the curve
	
	# if there is 2*threshold between the leftmost point in the curve and the left border
	# then the first partition occurs on domain['x_min'] + threshold
	if abs(domain['x_min'] - domain['x_min_cv']) >= 2* threshold :
		x_divs.append( domain['x_min'] + threshold  )
	# divide the curve in two
	x_divs.append( domain['center'][0] )
	
	# the next division is symmetrical to the previous one
	x_divs.append( x_divs[-1] + abs(x_divs[-1] - x_divs[-2]) )
	
	# the remaining of the domain is equally divided
	x_divs.append( np.linspace(start=x_divs[-1], stop=domain['x_max'], num=k - len(x_divs) +3)[1:] )
	x_divs = np.hstack(x_divs)
	
	# the convention of the borders on the curve, first half, is
	# the left border of the partition is the normal one and the right 
	# curve itself
	# the top is the top and the vertical part until it reaches the curve
	# and the bottom is the bottom and the vertical until it reaches the curve
	
	# the second half of the curve is the same, just changing the left for the right
	# and the rest of the partition as the square division, top is top, bottom is bottom, 
	# left is left and right is right
	
	# TODO have not done it yet
	# ATTENTION to the corners, remember to start some in the "second" point
	
	# to avoid roundoff errors of the index, reassign resolution
	resolution = int(resolution/2)*2
	
	# the order of the borders is top, bottom, left, right
	for i, xi in enumerate(x_divs[:-1]):
		# CHECK if we are dealing with the curve partition
		# Think I need to deal with each side separatedly	
		
		if (x_divs[i+1] == domain['center'][0]):
			# the first half of the curve
			
			# compute the number of points in the horizontal and vertical paths
			n_pts_horizontal = int((abs(x_divs[i+1] - xi)) / (abs(x_divs[i+1] - xi) + abs(domain['y_max'] - domain['y_max_cv'])) * resolution)
			n_pts_vertical = resolution - n_pts_horizontal
			
			top = (	np.hstack((np.linspace(xi, x_divs[i+1], n_pts_horizontal), [x_divs[i+1]]*n_pts_vertical)), 
					np.hstack((domain['y_max']*n_pts_horizontal, np.linspace(domain['y_max'], domain['y_max_cv'], n_pts_vertical))) )
			
			bottom = (np.hstack((np.linspace(xi, x_divs[i+1], n_pts_horizontal), [x_divs[i+1]]*n_pts_vertical)), 
						np.hstack((domain['y_min']*n_pts_horizontal, np.linspace(domain['y_min'], domain['y_min_cv'], n_pts_vertical))) )
			
			# since the curve starts at the rightmost point, I can start here with the
			# the point located at 1/4 of the curve length and go up until 3/4 
			left = (	domain['curve'][int(resolution/2):int(resolution*3/2)], 
						domain['curve'][int(resolution/2):int(resolution*3/2)])
			right =	([xi]*resolution, np.linspace(domain['y_max'], domain['y_min'], resolution))
			
			
		elif (xi == domain['center'][0]):
			# the second half of the curve
			
			# compute the number of points in the horizontal and vertical paths
			n_pts_horizontal = int((abs(x_divs[i+1] - xi)) / (abs(x_divs[i+1] - xi) + abs(domain['y_max'] - domain['y_max_cv'])) * resolution)
			n_pts_vertical = resolution - n_pts_horizontal
			
			top = (	np.hstack(([xi]*n_pts_vertical, np.linspace(xi, x_divs[i+1], n_pts_horizontal))), 
					np.hstack((np.linspace(domain['y_max_cv'], domain['y_max'], n_pts_vertical), domain['y_max']*n_pts_horizontal)) )
			
			bottom = (np.hstack(([xi]*n_pts_vertical, np.linspace(xi, x_divs[i+1], n_pts_horizontal))), 
						np.hstack((np.linspace(domain['y_min_cv'], domain['y_min'], n_pts_vertical), domain['y_min']*n_pts_horizontal)) )
			
			left = (	domain['curve'][int(resolution/2):-int(resolution*3/2)], 
						domain['curve'][int(resolution/2):-int(resolution*3/2)])
			right = ([x_divs[i+1]]*resolution, np.linspace(domain['y_max'], domain['y_min'], resolution))
		else:
			# suppose not
			top = (np.linspace(xi, x_divs[i+1], resolution), [domain['y_max']]*resolution)
			bottom = (np.linspace(xi, x_divs[i+1], resolution), [domain['y_min']]*resolution)
			left = ([xi]*resolution, np.linspace(domain['y_max'], domain['y_min'], resolution))
			right = ([x_divs[i+1]]*resolution, np.linspace(domain['y_max'], domain['y_min'], resolution))
#		top, bottom, left, right = [], [], [], []
		
		print(top)
		print()
		
		borders[i].append( [top, bottom, left, right] )
	
	
	
	return (x_divs, borders)

"""
# TEST
imp.reload(pjt); bb = pjt.heuristic_1(vv, 5, 50, 1)


k=0; [plt.plot(nn[k][1][i][0], nn[k][1][i][1]) for i in range(4)]; plt.show(); plt.close('all')

"""



def heuristic_2():
	"""
####
#
##
#
##
#

	"""
	# TODO think in a way to reduce this case into the previous one just adjusting something
	
	# the partitions are equally spaced in area
	# the first partition define how much area each should have

	# if it is not possible to have all partitions with the same area,
	# at least make the partition after the curve with the same principle
	# and then for the rest adjust its size to fit all k partitions
	
	
	# ATTENTION make the right border of the previous partition equal the left border
	# of the next partition
	
	# start by the division of the curve into two, then continue the partition
	
	
	pass


def partitionate_domain(domain, k, heuristic):
	"""
####
# 
###
# 
###
# 
	"""
	# depending on the space between the leftmost point of the curve to the left border
	# generate the first partition on the middle of the curve, without a "blank" partition
	# before the curve
	
	
	# ATTENTION the partition should have the same points on the interface, the connection, between them
	pass

