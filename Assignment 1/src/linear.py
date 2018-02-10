import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x = np.loadtxt('../data/linearX.csv').reshape(-1,1)
y = np.loadtxt('../data/linearY.csv').reshape(-1,1)

x = (x - np.mean(x))/np.std(x)                  # Normalize
x = np.concatenate((np.ones(x.shape),x),axis=1)

############ Part (a) ############

def gradient_descent(x,y,eta,maxit,mincost,verbose=False,theta_init=None):
    """
        Gradient Descent Implementation  
        Args:  
            x,y     :   Training Data  
            eta     :   Learning Rate  
            maxit   :   Maximum number of Iterations  
            mincost :   Minimum cost value  
            verbose :   Prints output regularly  
            theta_init (Optional): Initial parameter values  
        Returns:  
            theta_list  :   Parameter values after each update
            cost_lsit   :   Cost value after each update
    """
    theta = np.zeros((x.shape[1],1)) if theta_init == None else theta_init
    theta_list = [theta]
    diff = y - np.dot(x,theta)
    cost = 0.5*np.sum(diff**2)
    cost_list = [cost]
    iterations = 0

    while(cost > mincost and iterations < maxit):     # Stopping Criteria
        theta = theta + eta*np.dot(x.T,diff)
        theta_list.append(theta)
        diff = y - np.dot(x,theta)
        cost = 0.5*np.sum(diff**2)
        cost_list.append(cost)
        iterations += 1
        if (iterations % 10 == 0 and verbose): print(iterations," - ",cost)

    print("Final Cost - ",cost)
    print("Final Parameters - {0},{1}".format(theta[0],theta[1]))

    return theta_list, cost_list

theta_list, cost_list = gradient_descent(x,y,0.0015,101,0.0001,True)

# theta[0] = 0.9966201
# theta[1] = 0.0013402

##################################

############ Part (b) ############

sns.set()
sns.regplot(x[:,1],y.ravel(),fit_reg=False,marker='+')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta_list[-1][0] + theta_list[-1][1] * x_vals
plt.plot(x_vals,y_vals)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

##################################

############ Part (c) ############

sns.reset_orig()
xs = np.linspace(0,2,20)
ys = np.linspace(-1,1,20)
xs, ys = np.meshgrid(xs,ys)
y_repeat = np.repeat(y,xs.shape[0]*xs.shape[1])
diffs = y_repeat.reshape(y.shape[0],xs.shape[0],xs.shape[1]) - np.einsum('ij,jkl->ikl',x,np.array([xs,ys]))
zs = 0.5*np.sum(diffs**2,axis=0)
fig = plt.figure(figsize=(7,7))
ax = fig.gca(projection='3d')
ax.view_init(elev=50,azim=-72)
ax.plot_surface(xs,ys,zs,antialiased=True,cmap=cm.viridis)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel("Cost")
ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 15
x_vals = np.array([a[0] for a in theta_list])
y_vals = np.array([a[1] for a in theta_list])
z_vals = np.array(cost_list)
for idx in range(z_vals.shape[0]):
    if(idx % 10 == 0): print("Iteration - {0}, Cost - {1}".format(idx,z_vals[idx]))
    ax.plot(x_vals[idx],y_vals[idx],z_vals[idx],color='orange',marker='.')
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.1)

##################################

############ Part (d) ############

def contour_plot(xlim,ylim,eta,theta_list,cost_list,verbose=False):
    """
        Plot Contours and visualize Gradient Descent  
        Args:  
            xlim,ylim   :   Plot limits for parameters (displayed along axis)  
            eta         :   Learning Rate  
            theta_list  :   Parameter values after each update  
            cost_list   :   Cost value after each update  
            verbose     :   Prints output regularly  
        Returns:  
            None  
    """
    sns.set()
    x_vals = np.array([a[0] for a in theta_list])
    y_vals = np.array([a[1] for a in theta_list])
    z_vals = np.array(cost_list)
    xlim = xlim if x_vals.min() > xlim[0] and x_vals.max() < xlim[1] else (x_vals.min(),x_vals.max())
    ylim = ylim if y_vals.min() > ylim[0] and y_vals.max() < ylim[1] else (y_vals.min(),y_vals.max())
    xs = np.linspace(xlim[0],xlim[1],20)
    ys = np.linspace(ylim[0],ylim[1],20)
    xs, ys = np.meshgrid(xs,ys)
    y_repeat = np.repeat(y,xs.shape[0]*xs.shape[1])
    diffs = y_repeat.reshape(y.shape[0],xs.shape[0],xs.shape[1]) - np.einsum('ij,jkl->ikl',x,np.array([xs,ys]))
    zs = 0.5*np.sum(diffs**2,axis=0)
    fig = plt.figure(figsize=(7,7))
    ax = fig.gca()
    plt.contour(xs,ys,zs,cmap=cm.viridis)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.title("Linear Regression - Gradient Descent (Contours) - " + r'$\eta = {0}$'.format(eta))
    for idx in range(x_vals.shape[0]):
        if(idx % 10 == 0 and verbose): print("Iteration - {0}, Cost - {1}".format(idx,z_vals[idx]))
        ax.plot(x_vals[idx],y_vals[idx],color='orange',marker='.')
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

contour_plot((-0.4,2),(-1,1),0.0015,theta_list,cost_list)

##################################

############ Part (e) ############

eta_list = [0.001,0.005,0.009,0.013,0.017,0.021,0.025]

for eta in eta_list:
    theta_list, cost_list = gradient_descent(x,y,eta,101,0.0001)
    contour_plot((-0.4,2),(-1,1),eta,theta_list,cost_list)

##################################

while(len(plt.get_fignums()) != 0):
    close = input("Finished plotting, Enter q to close all (or figure number to close particular figure) - ")
    if(close == 'q' or close == 'Q'):
        plt.close('all')
    else:
        plt.close(int(close))