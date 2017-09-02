def grid2vtk(gx,gy,filename):
        n,m = gx.shape
        f = open(filename,'wt')

        f.write('# vtk DataFile Version 2.0\n')
        f.write('file generated by grid2vtk\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('POINTS '+str(n*m)+' float\n')
        for i in range(n):
            for j in range(m):
                f.write("{:.4f}".format(gx[i,j])+' '+"{:.4f}".format(gy[i,j])+' 0.0\n')

        f.write('\n')
        nc = (n-1)*(m-1)
        f.write('CELLS '+str(nc)+' '+str(nc*5)+'\n')
        for i in range(n-1):
            for j in range(m-1):
                f.write('4 '+str(i*m+j)+' '+str(i*m+j+1)+' '+str((i+1)*m+j+1)+' '+str((i+1)*m+j)+'\n')

        f.write('\n')
        f.write('CELL_TYPES '+str(nc)+'\n')
        f.write(nc*'7 ')
        f.write('\n')

        f.close()
