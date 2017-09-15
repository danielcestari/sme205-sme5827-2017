
# coding: utf-8

# In[24]:

import sys
import numpy as np
import matplotlib.pyplot as plt
from grid2vtk import grid2vtk
#get_ipython().magic('matplotlib inline')





def grid(filename, save_file, iter_number, xis_rf, etas_rf, points_rf, 
			a_xis, b_xis, c_xis, d_xis, 
			a_etas, b_etas, c_etas, d_etas, plot=False):

	f = open(filename,'rt')

	nx = int(f.readline())
	rt = np.zeros((nx,2))
	for i in range(nx):
		l=f.readline()
		rt[i,0]=l.split(' ')[0]
		rt[i,1]=l.split(' ')[1]

	n2 = int(f.readline())

	if (n2 != nx):
		print("top and botton discretization should match")
		sys.exit(0)

	rb = np.zeros((nx,2))
	for i in range(nx):
		l=f.readline()
		rb[i,0]=l.split(' ')[0]
		rb[i,1]=l.split(' ')[1]

	ny = int(f.readline())
	rl = np.zeros((ny,2))
	for i in range(ny):
		l=f.readline()
		rl[i,0]=l.split(' ')[0]
		rl[i,1]=l.split(' ')[1]

	n2 = int(f.readline())

	if (n2 != ny):
		print("left and right discretization should match")
		sys.exit(0)

	rr = np.zeros((ny,2))
	for i in range(ny):
		l=f.readline()
		rr[i,0]=l.split(' ')[0]
		rr[i,1]=l.split(' ')[1]

	f.close()

	gridx = np.zeros((ny,nx))
	gridy = np.zeros((ny,nx))

	gridx[0,:]=rb[:,0]
	gridx[ny-1,:]=rt[:,0]
	gridx[:,0]=rl[:,0]
	gridx[:,nx-1]=rr[:,0]

	gridy[0,:]=rb[:,1]
	gridy[ny-1,:]=rt[:,1]
	gridy[:,0]=rl[:,1]
	gridy[:,ny-1]=rr[:,1]

	dx = 1.0/nx
	dy = 1.0/ny
	for j in range(1,ny-1):
		for i in range(1,nx-1):
			idx = i*dx
			jdy = j*dy
			gridx[j,i] = (1.0-idx)*rl[j,0] + idx*rr[j,0] + (1.0 - jdy)*rb[i,0] + jdy*rt[i,0] - (1.0-idx)*(1.0-jdy)*rb[0,0] - (1.0 - idx)*jdy*rt[0,0] - idx*(1.0-jdy)*rb[nx-1,0] - idx*jdy*rt[nx-1,0]
			gridy[j,i] = (1.0-idx)*rl[j,1] + idx*rr[j,1] + (1.0 - jdy)*rb[i,1] + jdy*rt[i,1] - (1.0-idx)*(1.0-jdy)*rb[0,1] - (1.0 - idx)*jdy*rt[0,1] - idx*(1.0-jdy)*rb[nx-1,1] - idx*jdy*rt[nx-1,1]

	dxi = 1.0/nx
	deta = 1.0/ny
	N = iter_number


	# parametros para (airfoil)
	#			Gx = (g22*P*dxdxi+g11*Q*dxdeta)
	#			Gy = (g22*P*dydxi+g11*Q*dydeta)
	xi_rf = 2.0
	eta_rf = 0.5
	a_rf=0.000010
	b_rf=0.00001
	c_rf=0.0000001
	d_rf=0.0000001

	# parametros para (airfoil)
	#			Gx = g*(P*dxdxi+Q*dxdeta)
	#			Gy = g*(P*dydxi+Q*dydeta)
	xi_rf = 0.45
	eta_rf = 1.0
	a_rf=0.000040
	b_rf=0.000015
	c_rf=9.1
	d_rf=9.1

	#a_rf=0
	b_rf=0
	#c_rf=0
	d_rf=0

	# parametros para swan.txt
	#			Gx = g*(P*dxdxi+Q*dxdeta)
	#			Gy = g*(P*dydxi+Q*dydeta)
	"""
	xi_rf = 0.01
	eta_rf = 0.01
	a_rf=0.00150
	b_rf=0.00155
	c_rf=1
	d_rf=1
	"""

	for k in range(N):
		for j in range(1,ny-1):
			for i in range(1,nx-1):
				dxdxi = (gridx[j,i+1]-gridx[j,i-1])/(2.0*dxi)
				dydxi = (gridy[j,i+1]-gridy[j,i-1])/(2.0*dxi)
				dxdeta = (gridx[j+1,i]-gridx[j-1,i])/(2.0*deta)
				dydeta = (gridy[j+1,i]-gridy[j-1,i])/(2.0*deta)
				g11 = dxdxi**2 + dydxi**2
				g22 = dxdeta**2 + dydeta**2
				g12 = dxdxi*dxdeta + dydxi*dydeta
				a = 4.0*(deta**2)*g22
				b = 4.0*dxi*deta*g12
				c = 4.0*(dxi**2)*g11
				g = np.abs(g11*g22-g12**2)
				xixi_rf = i*dxi-xi_rf
				etaeta_rf = j*deta-eta_rf
				
				# first compute the P for the xis, then add to the sum the components regarding the point
				# the same goes for the Q
				P = sum([
							a_xis[m]*(i*dxi-xi_rf)/np.abs((i*dxi-xi_rf)+1.0e-2)*np.exp(-c_xis[m]*np.abs(i*dxi-xi_rf))
						for m, xi_rf in enumerate(xis_rf)])
				P += sum([
							b_xis[l]*(i*dxi-point_rf[0])/np.abs(i*dxi-point_rf[0])*np.exp(-d_xis[l]*np.sqrt((i*dxi-point_rf[0])**2+(j*deta-point_rf[1])**2))
						for l, point_rf in enumerate(points_rf)])
				
				Q = sum([
							a_etas[m]*(j*deta-eta_rf)/np.abs((j*deta-eta_rf)+1.0e-2)*np.exp(-c_etas[m]*np.abs(j*deta-eta_rf))
						for m, eta_rf in enumerate(etas_rf)])
				Q += sum([
							b_etas[l]*(j*deta-point_rf[1])/np.abs(j*deta-point_rf[1])*np.exp(-d_etas[l]*np.sqrt((i*dxi-point_rf[0])**2+(j*deta-point_rf[1])**2))
						for l, point_rf in enumerate(points_rf)])
	#			Q = a_rf*(etaeta_rf)/np.abs(etaeta_rf+1.0e-2)*np.exp(-c_rf*np.abs(etaeta_rf)) + b_rf*(etaeta_rf)/np.abs(etaeta_rf)*np.exp(-c_rf*np.sqrt(xixi_rf**2+etaeta_rf**2))
				Gx = g*(P*dxdxi+Q*dxdeta)
				Gy = g*(P*dydxi+Q*dydeta)
				#Gx = (g22*P*dxdxi+g11*Q*dxdeta)
				#Gy = (g22*P*dydxi+g11*Q*dydeta)
				gridx[j,i] = -1.0*Gx+(1.0/(2*(a+c)))*(a*(gridx[j,i+1]+gridx[j,i-1])+c*(gridx[j+1,i]+gridx[j-1,i]))-0.5*(b*(gridx[j+1,i+1]+gridx[j-1,i-1]-gridx[j+1,i-1]-gridx[j-1,i+1]))
				gridy[j,i] = -1.0*Gy+(1.0/(2*(a+c)))*(a*(gridy[j,i+1]+gridy[j,i-1])+c*(gridy[j+1,i]+gridy[j-1,i]))-0.5*(b*(gridy[j+1,i+1]+gridy[j-1,i-1]-gridy[j+1,i-1]-gridy[j-1,i+1]))

	grid2vtk(gridx,gridy, save_file)
	
	if plot:
		for i in range(nx):
			plt.plot(gridx[i,:],gridy[i,:],color='gray')
		for i in range(ny):
			plt.plot(gridx[:,i],gridy[:,i],color='gray')

		plt.plot(rt[:,0],rt[:,1])
		plt.plot(rb[:,0],rb[:,1])
		plt.plot(rl[:,0],rl[:,1])
		plt.plot(rr[:,0],rr[:,1])
		plt.show()
		plt.close('all')
	
	return (gridx, gridy)

# In[ ]:




# In[ ]:


def str_to_float_list(st):
	return [float(i) for i in st.split(',') if i]
def str_to_float_tuple_list(st):
	return [tuple(float(j) for j in i.split(',')) for i in st.split(';') if i]


if __name__ == '__main__':

	"""
examples:

# to run the practical exercise 2
# the default values are already set
python poisson.py

# input example, only controlling over xi on 3 lines, 0.2, 0.45, and 0.8
 python poisson.py airfoil.txt saida.txt 50 '0.2,0.45,0.8' '' '' '0.00002,0.00004,0.00002' '' '20,20,20' '' '' '' '' '' 

# concentrate lines over the point (0.5,0.5)
python poisson.py airfoil.txt saida.vtk 100 '' '' '0.5,0.5' '' '0.00006' '' '20' '' '0.0006' '' '20'
	"""	


	# the default values works for the practical exercise two
	
	filename = 'airfoil.txt' if len(sys.argv) < 2 else sys.argv[1]
	save_file = 'output.vtk' if len(sys.argv) < 3 else sys.argv[2]
	iter_number = 100 if len(sys.argv) < 4 else int(sys.argv[3])

	# control xis, etas, and points
	xis_rf = [0.45] if len(sys.argv) < 5 else str_to_float_list(sys.argv[4])
	etas_rf = [1.0] if len(sys.argv) < 6 else str_to_float_list(sys.argv[5])
	points_rf = [(0.45,1.0)] if len(sys.argv) < 7 else str_to_float_tuple_list(sys.argv[6])

	# control parameters for every xi, eta, and point
	a_xis = [0.00002] if len(sys.argv) < 8 else str_to_float_list(sys.argv[7])
	b_xis = [0.00002] if len(sys.argv) < 9 else str_to_float_list(sys.argv[8])
	c_xis = [20] if len(sys.argv) < 10 else str_to_float_list(sys.argv[9])
	d_xis = [20] if len(sys.argv) < 11 else str_to_float_list(sys.argv[10])

	a_etas = [0.00004] if len(sys.argv) < 12 else str_to_float_list(sys.argv[11])
	b_etas = [0.00004] if len(sys.argv) < 13 else str_to_float_list(sys.argv[12])
	c_etas = [20] if len(sys.argv) < 14 else str_to_float_list(sys.argv[13])
	d_etas = [20] if len(sys.argv) < 15 else str_to_float_list(sys.argv[14])

	
	print(('filename', filename, 'save_file', save_file, 'iter_number', iter_number,))
	print(('xis_rf', xis_rf, 'etas_rf', etas_rf, 'points_rf', points_rf, ))
	print(('a_xis', a_xis, 'b_xis', b_xis, 'c_xis', c_xis, 'd_xis', d_xis, ))
	print(('a_etas', a_etas, 'b_etas', b_etas, 'c_etas', c_etas, 'd_etas', d_etas, ))

	grid(filename=filename, save_file=save_file, iter_number=iter_number, xis_rf=xis_rf, etas_rf=etas_rf, 
			points_rf=points_rf, a_xis=a_xis, b_xis=b_xis, c_xis=c_xis, d_xis=d_xis, 
			a_etas=a_etas, b_etas=b_etas, c_etas=c_etas, d_etas=d_etas, plot=True)
