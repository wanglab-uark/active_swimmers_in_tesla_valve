####################################################
# Simulation for Active Swimmers (Active Brownian Motion Particles)
# Author: Wang, Rogers, and He.
# This repository includes the essential code used in the manuscript: 
#       Rogers, He, and Wang. ".........."
# The algorithms are based on a published paper by Volpe et al.
#       Volpe G, Gigan S, Volpe G (2014) Simulation of the active Brownian motion of a microswimmer. American Journal of Physics, 82(7):659–664. https://doi.org/10.1119/1.4870398
# All Rights Reserved.
####################################################

# %% import packages
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import skimage.measure
import skimage.segmentation
import skimage.morphology
import pims
from scipy import ndimage as ndi

# %% define functions
def polyarea(x, y):
    ''' calculate the area of a polygon (specified by vertices x, y) using the Shoelace formula
    Credit: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    '''
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def inpolygon(point, polygon, radius=0.0):
    '''Check if a point is inside a polygon.
    :param point: [x, y], point
    :param polygon: list of vertices of the polygon
    :param radius: radius of the edge of the polygon (or equivalently the radius of the point)
    :return: True if the point is inside the polygon, False otherwise
    '''
    p = mpl.path.Path(polygon)
    return p.contains_point(point, radius=radius)

def get_reflection(ri, rf, boundary):
    '''Get the reflected position of a point at a boundary.
    :param ri: [xi, yi], initial position before colliding with the boundary
    :param rf: [xf, yf], "trial" final position inside the boundary
    :param boundary: list of vertices of the boundary
    :return: rr=[xr, yr], reflected position of the point after reflection at the boundary
    '''
    # get the line segment (y=ax+b) connecting ri and rf
    NL = 1000 # number of points of the line segment
    xi, yi = ri
    xf, yf = rf
    if xi==xf: # vertical line (same x, different y)
        xl = np.ones(NL)*xi
        yl = np.linspace(yi, yf, NL)
    else: # get the slope and intercept, and generate the line
        a = (yf-yi)/(xf-xi)
        b = yi - a*xi
        xl = np.linspace(xi, xf, NL)
        yl = a*xl + b
    
    # determine the intersection of the line with the boundary
    xv = boundary[:, 0]
    yv = boundary[:, 1]
    NV = len(xv)
    # get pair-wise distances (slow version)
    # dist = np.zeros((NL, NV))
    # for i in range(NL):
    #     for j in range(NV):
    #         dist[i, j] = np.sqrt((xl[i]-xv[j])**2 + (yl[i]-yv[j])**2)
    # get pair-wise distances (fast version)
    mxl = np.tile(xl,(NV,1)).T
    myl = np.tile(yl,(NV,1)).T
    mxv = np.tile(xv,(NL,1))
    myv = np.tile(yv,(NL,1))
    dist = np.sqrt((mxl-mxv)**2 + (myl-myv)**2)
    idxlp, idxbp = np.unravel_index(np.argmin(dist, axis=None), dist.shape) # idxbp is the index of the closes point from the boundary
    # if idxbp is the last vertex, then the next index should be the start 
    if idxbp==NV-1:
        nextbp = 0
    else:
        nextbp = idxbp+1
    
    # get the reflected position
    if xv[nextbp]==xv[idxbp]: # vertical boundary at the intersection point -> reflection is along the y-axis
        rr = [2*xv[idxbp]-xf, yf]
    else: # not a vertical boundary -> need to get the tangent line (slope and intercept
        m = (yv[nextbp]-yv[idxbp])/(xv[nextbp]-xv[idxbp]) # slope of the tangent line
        c = yv[idxbp] - m*xv[idxbp] # intercept of the tangent line
        # get the reflected position (https://math.stackexchange.com/questions/1013230/how-to-find-coordinates-of-reflected-point)
        a, b = m, -1
        tmp = -2.0*(a*xv[idxbp]+b*yv[idxbp]+c)/(a**2+b**2)
        rr = [xv[idxbp]+a*tmp, yv[idxbp]+b*tmp]

    # return
    return np.array(rr)

def bac_swim_POMIB(N, dt, t0, r0, theta0, R, T, eta, V, W, 
    period_boundary, inner_boundaries, message, verbose=False):
    '''
    Simulate the swimming of a single bacterium without POMIB (periodic outer boundary, multile inner boundaries).

    :param N: number of time points
    :param dt: time step
    :param t0: initial time (first step)
    :param r0: [x0, y0], initial position of the bacterium
    :param theta0: initial angle of the bacterium
    :param R: radius of the bacteria in m (e.g., 0.8e-6 m)
    :param T: temperaure in K
    :param eta: viscosity in Pa s (e.g., 1e-3 for water)
    :param V: (linear) swimming speed in m/s
    :param W: angular speed in rad/s
    :param period_boundary: [xmin, xmax, ymin, ymax], rectangular periodic boundary. When the bacterium goes >xmax (or ymax), it will re-appear at xmin (or ymin); vice versa.
    :param inner_boundaries: list of inner boundaries (obstacles). Each boundary contains the positions of the vertices (array-[xv,yv]).
    :param message: message for logging purpose (such as bacteria/particle ID, trial ID, etc.)
    :param verbose: True if want to print out data/info
    :return: (r, t, DT, DR)
        r: position / trajectory
        t: time points
        DT: translational diffusion coefficient
        DR: rotational diffusion coefficient
    '''
    # constants 
    kB = 1.38e-23       # Boltzmann constant [J/K]
    gamma = 6*np.pi*R*eta  # friction coefficient [Ns/m]
    DT = kB*T/gamma     # translational diffusion coefficient [m^2/s]
    DR = 6*DT/(8*R**2)   # rotational diffusion coefficient [rad^2/s]

    # initialize variables
    t = np.arange(t0, t0+(N+1)*dt, dt)
    r = np.zeros((N+1, 2))
    r[0, :] = r0
    theta = theta0      # initial angle
    xmin, xmax, ymin, ymax = period_boundary # periodic boundary

    # simulation for each step
    for i in range(N):
        if i%1000==0:
            print('[%s]: INFO: Step %d/%d ...'%(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], i, N))
        isGood = False
        rp = r[i, :]
        curtheta = theta
        while not isGood:
            # take a translational diffusionstep
            rt = rp + np.sqrt(2*DT*dt)*np.random.randn(2)
            # take a rotional diffusion step
            theta = theta + np.sqrt(2*DR*dt)*np.random.randn()
            # take a torque/rotation step
            theta = theta + dt*W
            # take a drift/transition step
            rt = rt + dt*V*np.array([np.cos(theta), np.sin(theta)])
            # check if the new position is outside the periodic boundary
            if rt[0] < xmin:
                rt[0] = xmax - (xmin - rt[0])
            elif rt[0] > xmax:
                rt[0] = xmin + (rt[0] - xmax)
            else:
                pass
            if rt[1] < ymin:
                rt[1] = ymax - (ymin - rt[1])
            elif rt[1] > ymax:
                rt[1] = ymin + (rt[1] - ymax)
            else:
                pass
            # check if the new position is inside the inner boundaries
            idx_of_inner_boundaries = -1
            for j in range(len(inner_boundaries)):
                obstacle = inner_boundaries[j]
                if inpolygon(rt, obstacle): # collision happened
                    idx_of_inner_boundaries = j
                    break
            if idx_of_inner_boundaries < 0: # no collision with the inner boundaries
                isGood = True
            else: # collision happened
                obstacle = inner_boundaries[idx_of_inner_boundaries]
                # reflect
                rt = get_reflection(rp, rt, obstacle)
                if not inpolygon(rt, obstacle):
                    isGood = True
        # update the position
        r[i+1, :] = rt
        if verbose:
            print('[DATA] %s %d %d %e %e %e %e %e %e'%(message, i, t[i], rp[0], rp[1], rt[0], rt[1], curtheta, theta))
    # return
    return (r, t, DT, DR)

def bac_swim_POSIB(N, dt, t0, r0, theta0, R, T, eta, V, W, 
    period_boundary, boundary, message, verbose=False):
    '''
    Simulate the swimming of a single bacterium without POSIB (periodic outer boundary, single inner boundary). See doc for `bac_swim_POMIB` for details.
    '''
    return bac_swim_POMIB(N, dt, t0, r0, theta0, R, T, eta, V, W, period_boundary, [boundary], message, verbose)

def bac_swim_SOMIB(N, dt, t0, r0, theta0, R, T, eta, V, W, 
    outer_boundary, inner_boundaries, message, verbose=False):
    '''
    Simulate the swimming of a single bacterium without SOMIB (single outer boundary, multile inner boundaries).

    :param N: number of time points
    :param dt: time step
    :param t0: initial time (first step)
    :param r0: [x0, y0], initial position of the bacterium
    :param theta0: initial angle of the bacterium
    :param R: radius of the bacteria in m (e.g., 0.8e-6 m)
    :param T: temperaure in K
    :param eta: viscosity in Pa s (e.g., 1e-3 for water)
    :param V: (linear) swimming speed in m/s
    :param W: angular speed in rad/s
    :param outer_boundary: outer boundary (envelope)
    :param inner_boundaries: list of inner boundaries (obstacles). Each boundary contains the positions of the vertices (array-[xv,yv]).
    :param message: message for logging purpose (such as bacteria/particle ID, trial ID, etc.)
    :param verbose: True if want to print out data/info
    :return: (r, t, DT, DR)
        r: position / trajectory
        t: time points
        DT: translational diffusion coefficient
        DR: rotational diffusion coefficient
    '''
    # constants 
    kB = 1.38e-23       # Boltzmann constant [J/K]
    gamma = 6*np.pi*R*eta  # friction coefficient [Ns/m]
    DT = kB*T/gamma     # translational diffusion coefficient [m^2/s], DT~1/R
    DR = 6*DT/(8*R**2)   # rotational diffusion coefficient [rad^2/s], DR~1/R^3

    # initialize variables
    t = np.arange(t0, t0+(N+1)*dt, dt)
    r = np.zeros((N+1, 2))
    r[0, :] = r0
    theta = theta0      # initial angle

    # simulation for each step
    for i in range(N):
        if i%1000==0:
            print('[%s]: INFO: Step %d/%d ...'%(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], i, N))
        isGood = False
        rp = r[i, :]
        curtheta = theta
        while not isGood:
            # take a translational diffusion step
            rt = rp + np.sqrt(2*DT*dt)*np.random.randn(2)
            # take a rotional diffusion step
            theta = theta + np.sqrt(2*DR*dt)*np.random.randn()
            # take a torque/rotation step
            theta = theta + dt*W
            # take a drift/transition step
            rt = rt + dt*V*np.array([np.cos(theta), np.sin(theta)])
            # check if the new position is inside the outer boundary (envelope)
            if not inpolygon(rt, outer_boundary): # outside the outer boundary, collision happened
                # reflect
                rt = get_reflection(rp, rt, outer_boundary)
                # check again
                if inpolygon(rt, outer_boundary): # inside the outer boundary, good (assume steps are small so that it does not collide with the inner boundaries)
                    isGood = True
                    break
            else: # no collision with the outer boundary
                # check if the new position is inside the inner boundaries
                idx_of_inner_boundaries = -1
                for j in range(len(inner_boundaries)):
                    obstacle = inner_boundaries[j]
                    if inpolygon(rt, obstacle): # collision happened
                        idx_of_inner_boundaries = j
                        break
                if idx_of_inner_boundaries < 0: # no collision with the inner boundaries
                    isGood = True
                else: # collision happened
                    obstacle = inner_boundaries[idx_of_inner_boundaries]
                    # reflect
                    rt = get_reflection(rp, rt, obstacle)
                    if not inpolygon(rt, obstacle):
                        isGood = True
        # update the position
        r[i+1, :] = rt
        if verbose:
            print('[DATA] %s %d %d %e %e %e %e %e %e'%(message, i, t[i], rp[0], rp[1], rt[0], rt[1], curtheta, theta))
    # return
    return (r, t, DT, DR)


def bac_swim_SOSIB(N, dt, t0, r0, theta0, R, T, eta, V, W, 
    outer_boundary, boundary, message, verbose=False):
    '''
    Simulate the swimming of a single bacterium without SOSIB (single outer boundary, single inner boundary). See doc for `bac_swim_SOMIB` for details.
    '''
    return bac_swim_SOMIB(N, dt, t0, r0, theta0, R, T, eta, V, W, outer_boundary, [boundary], message, verbose)

def get_boundaries(boundary_img, threshold=100, minsize=10):
    # skimage.measure.find_contours
    if type(boundary_img) == str:
        boundary_img = pims.open('tesla_valve_yw01_filled.png')[0]
    if len(boundary_img.shape) >= 3:
        boundary_img = boundary_img[:,:,0]
    bwimg = np.asarray(boundary_img<threshold).astype(int)
    markers = ndi.label(bwimg)[0]
    labels = skimage.segmentation.watershed(np.zeros_like(bwimg), markers, mask=bwimg)
    edges = skimage.measure.find_contours(labels, 0)
    boundaries = []
    for ed in edges:
        if polyarea(ed[:,0], ed[:,1])>minsize and len(ed[:,0])>5:
            boundaries.append(np.array([ed[:,1], ed[:,0]]).T) # x,y should be switched
    return boundaries

def rescale_boundaries(boundaries, maxby='width', maxval=100e-6):
    '''rescale (and translate) the boundaries so that the boundaries are within the range [0, maxval] by `maxby` (width/x or height/y)
    '''
    new_boundaries = []
    xmin = np.min([np.min(b[:,0]) for b in boundaries])
    xmax = np.max([np.max(b[:,0]) for b in boundaries])
    ymin = np.min([np.min(b[:,1]) for b in boundaries])
    ymax = np.max([np.max(b[:,1]) for b in boundaries])
    dx = xmax - xmin
    dy = ymax - ymin
    if maxby == 'width':
        factor = maxval/np.max(dx)
    else:
        factor = maxval/np.max(dy)
    for b in boundaries:
        new_boundaries.append(np.array([(b[:,0]-xmin)*factor, (b[:,1]-ymin)*factor]).T)
    return new_boundaries

def get_SOMIB(boundary_img, maxby='width', maxval=100e-6, threshold=100, minsize=100e-12):
    boundaries = get_boundaries(boundary_img, threshold, minsize)
    boundaries = rescale_boundaries(boundaries, maxby, maxval)
    if len(boundaries)<2: # not enough boundaries
        return (None, boundaries[0])
    areas = [polyarea(bd[:,0], bd[:,1]) for bd in boundaries]
    idx = np.argmax(areas)
    outer_boundary = boundaries[idx]
    inner_boundaries = []
    for i in range(len(boundaries)):
        if i != idx:
            inner_boundaries.append(boundaries[i])
    return (outer_boundary, inner_boundaries)

def plotr(r, ls='-', **kwargs):
    '''plot results'''
    plt.plot(r[:,0], r[:,1], ls, **kwargs)

# %% run simulation and plot results
def main():
    bo, bis = get_SOMIB('tesla_valve_yw01_filled.png', maxby='height', maxval=40e-6, threshold=100, minsize=100e-12)
    if os.path.exists('test_tesla_valve_single_bac.pkl'):
        print('[%s]: loading simulation results ...'%datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        r, t, DT, DR = pickle.load(open('test_tesla_valve_single_bac.pkl', 'rb'))
        print('[%s]: DONE!'%datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    else:
        print('[%s]: Starting simulation ...'%datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        res = bac_swim_SOMIB(100000, 0.02, 0, np.array([220, 20])*1e-6, 0, 1e-6, 300, 1e-3, 10e-6, 0, bo, bis, message='', verbose=False)
        # parameters:  N, dt, t0, r0, theta0, R, T, eta, V, W, outer_boundary, inner_boundaries, message, verbose=False
        print('[%s]: DONE!'%datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        pickle.dump(res, open('test_tesla_valve_single_bac.pkl', 'wb'))
        r, t, DT, DR = res
    plt.figure(figsize=(10, 10), dpi=300)
    plotr(bo, 'k-', lw=2)
    for ib in bis:
        plotr(ib, 'y-', lw=1)
    plotr(r, 'b-', lw=1)
    plt.plot(r[0,0], r[0,1], 'g>')
    plt.plot(r[-1,0], r[-1,1], 'rs')
    plt.gca().set_aspect('equal')
    plt.savefig('fig_res.png', dpi=300)
    plt.show()

if __name__=="__main__":
    main()
