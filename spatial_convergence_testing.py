import h5py
import numpy as np
import matplotlib.pyplot as plt

# these lines may or may not be necessary to show plots
import matplotlib as mpl
mpl.use('tkagg')

# read out velocity data from file and compensate for staggering 
def getVelField(x,y,dataSet,dataName):
    
    # read out 
    thisData = dataSet.get(dataName)
    u = thisData.get('XVelocity')
    v = thisData.get('ZVelocity')
    
    # first frame in j
    u = u[:,0,:]
    v = v[:,0,:]
    
    # averaging
    u = (u + np.roll(u,-1,axis = 0))/2
    v = (v + np.roll(v,-1,axis = 1))/2
    
    # delete extra point used in differncing 
    N = int(np.sqrt(u.size))
    
    u = np.delete(u,N-1,axis = 0)
    u = np.delete(u,N-1,axis = 1)
    v = np.delete(v,N-1,axis = 0)
    v = np.delete(v,N-1,axis = 1)
    
    # delete extra point used in differencing 
    xvel = x[:N-1]
    yvel = y[:N-1]
    
    return u,v,xvel,yvel

# read out 2D grid points
def twoDGrid(dataSet):
    # read out
    grid = dataSet.get('GRID_DATA')
    x = grid.get('d0_gridpoints')[:]
    y = grid.get('d2_gridpoints')[:]
    
    return x,y

def twoDVortCalc(x,y,u,v):
    # find dx, dy
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # calculate dv/dx, du/dy by difference methods
    dvdx = (np.roll(v,-1,axis = 1) - np.roll(v,1,axis=1))/(2*dx)
    dudy = (np.roll(u,-1,axis=0) - np.roll(u,1,axis=0))/(2*dy)
    
    vort = dvdx - dudy
    
    # delete extra points
    M = vort.shape[0]
    vort = vort[1:M-1,1:M-1]
    
    # delete extra point for vorticity
    xvort = x[1:M-1]
    yvort = y[1:M-1]
    
    return vort,xvort,yvort

def calc_convx(param_8del, param_4del, param_2del, param_del,x,y):
    [Nx,Ny] = param_8del.shape
    
    reduced4 = param_4del[1::2,1::2]
    reduced2 = param_2del[3::4,3::4]
    reduced = param_del[7::8,7::8]
    
    w0 = 100
    kx = 2*np.pi
    ky = 2*np.pi
    k_sqrd = kx**2 + ky**2
    nu = 1/400
    
    dx_21 = x[1] - x[0]
    dx_42 = 2*dx_21
    dx_84 = 4*dx_21
    
    error_84 = param_8del - reduced4
    error_42 = reduced4 - reduced2
    error_21 = reduced2 - reduced
        
    analytical = 0#1
    
    if analytical:
        U = np.empty([len(x),len(y)])
        for i in range(len(x)):
            xg = x[i]
            for j in range(len(y)):
                yg = y[j]
                U[j,i] = (-w0*ky/k_sqrd)*np.cos(kx*xg)*np.sin(ky*yg)*np.exp(-nu*k_sqrd*.25)
        plt.figure()
        plt.contourf(param_8del-U)
        plt.colorbar()
        plt.figure()
        plt.contourf(reduced4-U)
        plt.colorbar()
        plt.figure()
        plt.contourf(reduced2-U)
        plt.colorbar()
        plt.figure()
        plt.contourf(reduced-U)
        plt.colorbar()
    else:
        plt.figure()
        plt.contourf(error_84)
        plt.colorbar()
        plt.figure()
        plt.contourf(error_42)
        plt.colorbar()
        plt.figure()
        plt.contourf(error_21)
        plt.colorbar()
    
    L2_norm84 = np.sqrt(np.sum(np.sum(error_84**2,axis=0))/(Nx*Ny))
    L2_norm42 = np.sqrt(np.sum(np.sum(error_42**2,axis=0))/(Nx*Ny))
    L2_norm21 = np.sqrt(np.sum(np.sum(error_21**2,axis=0))/(Nx*Ny))
    
    rate1 = L2_norm84/L2_norm42
    rate2 = L2_norm42/L2_norm21
    order = (np.log(rate1) + np.log(rate2))/2/np.log(2)
    
    plt.figure()
    plt.loglog([dx_21,dx_42,dx_84],[L2_norm21,L2_norm42,L2_norm84])
    plt.loglog([dx_21,dx_42,dx_84],[L2_norm21,L2_norm42,L2_norm84],'ro')
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$E$")
    plt.show()
    
    return order, L2_norm21,L2_norm42,L2_norm84

# read through the last frame and output desired parameter
def readTimeFrames(dataSet,Pr,plot,conv):
    # constants from sim
    w0 = 100
    kx = 2*np.pi
    ky = 2*np.pi
    k_sqrd = kx**2 + ky**2
    nu = 1/400
    
    # read out time and grid data
    times = dataSet.get('times')[:]
    #print(times)
    x,y = twoDGrid(dataSet)
    
    # for each instance 
    i = 0
    for dataName in dataSet.keys():
        if i < len(list(dataSet.keys()))-3:

            if i == len(list(dataSet.keys()))-4:
                # read out velocity data
                u,v,xvel,yvel = getVelField(x,y,dataSet,dataName)

                # read out pressure data
                #if Pr:
                #    P = pressureField(dataSet,dataName)
                
                if plot:
                    # calculate 2D vorticity in z direction
                    vort,xvort,yvort = twoDVortCalc(x,y,u,v)
                
                    #  compare vorticity with analytical solution at some y
                    y0 = 0.5 # should be mupltiple of .03125 for this sim (32x32)
                    anVort = w0*np.cos(kx*(xvort))*np.cos((y0)*ky)*np.exp(-nu*k_sqrd*times[i])
                    ind = np.argwhere(yvort == y0)[0]
                    oneDVort = vort[ind,:]

                    # plot flow field
                    fig = plt.figure(figsize = [20,10])
                    plt.quiver(xvel,yvel,u,v)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlabel("x",fontsize=12)
                    plt.ylabel("y",fontsize=12)
                    #plt.show()
                    #plt.grid(which='both',axis='both')
                    #plt.savefig('tgv_vel_field_vel_averaging.pdf',bbox_inches = 'tight')
                                 
                    # plot 1D vorticity
                    plt.figure(figsize = (20,5))
                    plt.subplot(1,3,1)
                    plt.plot(xvort,oneDVort[0,:])
                    plt.xlabel("x",fontsize=12)
                    plt.ylabel(r"$\omega_z$",fontsize=12)           
                    # plot analytical vorticity
                    plt.plot(xvort,anVort)
                    plt.legend(["Simulated vorticity","Analytical solution"],fontsize=12)
                    #plt.show()
                
                    # plot 1D u
                    plt.subplot(1,3,2)
                    plt.plot(xvel,u[ind+1,:][0])
                    #print(w0*ky/k_sqrd)
                    plt.plot(xvel,(-w0*ky/k_sqrd)*np.cos((xvel)*kx)*np.sin((y0)*ky)*np.exp(-nu*k_sqrd*times[i]))
                    plt.xlabel("x",fontsize=12)
                    plt.ylabel("u",fontsize=12)
                    plt.legend(["Simulated","Analytical solution"],fontsize=12)
                    #plt.show()
                
                    # plot 1D v
                    plt.subplot(1,3,3)
                    plt.plot(xvel,v[ind+1,:][0])
                    plt.plot(xvel,(w0*kx/k_sqrd)*np.sin((xvel)*kx)*np.cos((y0)*ky)*np.exp(-nu*k_sqrd*times[i]))  
                    plt.xlabel("x",fontsize=12)
                    plt.ylabel("v",fontsize=12)
                    plt.legend(["Simulated","Analytical solution"],fontsize=12)
                    #plt.show()
                
                    #plt.savefig('tgv_vorticity.pdf',bbox_inches = 'tight')
                
                    # if Pr:
                    #     # 2D pressure contour
                    #     plt.figure(figsize = (10,10))
                    #     plt.contourf(x, y, P)
                    #     plt.gca().set_aspect('equal', adjustable='box')
                    #     plt.xlabel("x",fontsize=12)
                    #     plt.ylabel("y",fontsize=12)
                    #     plt.colorbar()
                    #plt.savefig('tgv_pressure.pdf',bbox_inches = 'tight')
                
                    # 2D calculated vorticity contour
                    plt.figure(figsize = (10,10))
                    #plt.contourf(xp, yp, Oz)
                    plt.contourf(xvort,yvort,vort)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlabel("x",fontsize=12)
                    plt.ylabel("y",fontsize=12)
                    plt.colorbar()
                    #plt.show()
                    #plt.savefig('tgv_omegaZ.pdf',bbox_inches = 'tight')
                
                    ## 2D exported vorticity contour
                    #plt.figure(figsize = (10,10))
                    #plt.contourf(xp, yp, Oz)
                    #plt.gca().set_aspect('equal', adjustable='box')
                    #plt.xlabel("x",fontsize=12)
                    #plt.ylabel("y",fontsize=12)
                    #plt.colorbar()
                    #plt.savefig('tgv_omegaZ.pdf',bbox_inches = 'tight')


            
                if i == len(list(dataSet.keys()))-4:
            
                    if len(conv) > 0:
                        if conv == "u":
                            return u, xvel, yvel
                        elif conv == "v":
                            return v, xvel, yvel
                        #elif conv == "P":
                        #    return P, x, y
                    else:
                        return
                
        i = i+1

f = h5py.File('VIZ_8dx.h5dns', 'r')
data = f.get('FIELD_SEQUENCE_field3d')
u8dx,x,y = readTimeFrames(data,0,0,"u")

f = h5py.File('VIZ_4dx.h5dns', 'r')
data = f.get('FIELD_SEQUENCE_field3d')
u4dx,_,_ = readTimeFrames(data,0,0,"u")

f = h5py.File('VIZ_2dx.h5dns', 'r')
data = f.get('FIELD_SEQUENCE_field3d')
u2dx,_,_ = readTimeFrames(data,0,0,"u")

f = h5py.File('VIZ_dx.h5dns', 'r')
data = f.get('FIELD_SEQUENCE_field3d')
udx,_,_ = readTimeFrames(data,0,0,"u")

cr_x = calc_convx(u8dx, u4dx, u2dx, udx,x,y)
print(cr_x)
#plt.savefig('x_conv.pdf',bbox_inches = 'tight')