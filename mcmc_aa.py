#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:02:58 2023

@author: tga
"""
import sys
sys.path.append(r'/opt/anaconda3/lib/python3.6/site-packages/emcee-0.0.0-py3.6.egg/')
import secrets
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp2d
import time
import csv
from scipy.interpolate import griddata
import emcee
from multiprocessing import Pool
import os
import math
import pandas as pd
from scipy import signal

##################### MCMC parameters ###########################

#pl = np.array([-1, -1,    0, -5,  0.15,   1, 1,  0, 27, 0.2,  8, -25])
#pu = np.array([ 1,  1, 6.28,  0,  0.40, 100, 3, 90, 29, 0.8, 10, -20])
pl = np.array([-2.5, -2, 10, -1, -1, -np.pi, -5,  0.1, 1,  0, 27, 0.1,  1, -25])
pu = np.array([-1.5, -1, 99,  1,  1,  np.pi,  0,  1.0, 3, 90, 29, 0.9, 10, -20])
ndim = len(pl)
nwalkers = 2 *ndim
nstep = 5000
nburn = nstep // 5
#par = np.array([0.1, 0.1, 4, -2, 0.25, 3, 2, 10, 28, 0.3, 9, -22])*(1+0.001*np.random.randn(nwalkers,ndim))
par = np.array([-2, -1.45, 33, 0.2, -0.7, 0.3, -1.1, 0.45, 1.3, 23.3, 28.2, 0.5, 8.3, -22.75])*(1+0.001*np.random.randn(nwalkers,ndim))
#par = np.random.uniform(pl,pu,(nwalkers,ndim))
#################  some settings  ##############
theta = np.linspace(1,100,99)/10  # deg
zeta = np.arange(180)*2*np.pi/180  # rad
#ev = np.logspace(13.4,15,17,endpoint=True)  # eV
ev = np.logspace(13.4,14,7,endpoint=True)  # eV
erg = ev * 1.6e-12
ifort="/opt/intel/oneapi/compiler/2023.0.0/linux/bin/intel64/ifort"
#################  functions  ################
def lnpoisson(m,n):
    if(m!=0 and n!=0):
        A = math.gamma(n+1)  #n!
        p = n*math.log(m)-m-math.log(A)
    elif(m!=0 and n==0):
        p = -m
    elif(m==0 and n!=0):
        A = math.gamma(n+1)
        p = n*math.log(0.0001)-0.0001-math.log(A)
    elif(m==0 and n==0):
        p = 0
    return p

def coordinate_transform(data_ra,data_dec,src_ra,src_dec):
    d_lon = data_ra*np.pi/180   #data ra
    d_lat = data_dec*np.pi/180  #data dec
    s_lon = src_ra*np.pi/180    #src ra
    s_lat = src_dec*np.pi/180      #src dec
#transform ra,dec to x,y,z on the unit sphere
    x = np.cos(d_lat)*np.cos(d_lon)
    y = np.cos(d_lat)*np.sin(d_lon)
    z = np.sin(d_lat)
    pos = np.mat([x,y,z])
#ratation matrix
    theta1 = np.pi/2+s_lon
    Rz1 = np.mat([[np.cos(theta1),np.sin(theta1),0],[-np.sin(theta1),np.cos(theta1),0],[0,0,1]])
    theta2 = np.pi/2-s_lat
    Rx = np.mat([[1,0,0],[0,np.cos(theta2),np.sin(theta2)],[0,-np.sin(theta2),np.cos(theta2)]])
    rotation_matrix = Rx*Rz1
#pos1 is the transformed position
    pos1 = rotation_matrix*pos
#transform x1,y1,z1 to ra1,dec1
    ra1 = np.array(np.arctan(pos1[0]/pos1[2]))*180/np.pi
    dec1 = np.array(np.arcsin(pos1[1]))*180/np.pi
    return ra1[0], dec1[0]

def anistropy_analytical(par):
    src_x, src_y, xi, eta, MA, s, phi, D0, delta, B0 = par
    name = secrets.token_hex(8)
    code = open(r"J0543_aa.f90",'r')
    script = code.readlines()
    code.close()
    script[27] = 'filename=\'' + name + '\'\n'
    script[28] = 'eta=10**' + str(eta)[:8] + '\n'
    script[29] = 'MA=' + str(MA)[:8] + '\n'
    script[30] = 's=' + str(s)[:8] + 'd0\n'
    script[31] = 'phi=' + str(phi)[:8] + '/180d0*pi\n'
    script[32] = 'D0=10**' + str(D0)[:8] + '\n'
    script[33] = 'delta=' + str(delta)[:8] + '\n'
    script[34] = 'B0=' + str(B0)[:8] + 'd-6\n'
#    script[35] = 'gmax=10**' + str(gmax)[:8] + '\n'
    f = open(name + '.f90','w')
    for s in script:
        f.write(s)
    f.close()
    os.system(ifort + ' -qopenmp -o ' + name + ' ' + name + '.f90')
    t5 = time.time()
    os.system(r'./' + name)
    t6 = time.time()
    os.system(r'rm ' + name)
    os.system(r'rm ' + name + '.f90')
    data = np.loadtxt(name + '.dat')
    data = data.reshape(99,91,7)
    os.system(r'rm ' + name + '.dat')
    dNdE = data / erg[None,None,:]**2 # 1 / erg cm**2 s
    N_counts = np.sum((dNdE[:,:,1:] + dNdE[:,:,:-1])*np.diff(erg)/2,axis=2) # 1 / cm**2 s
    NN_counts = np.zeros((99,180))
    NN_counts[:,:91] = N_counts
    NN_counts[:,91:] = N_counts[:,89:0:-1] 
    x = theta[:,None]*np.cos(zeta[None,:] + xi) + src_x
    y = theta[:,None]*np.sin(zeta[None,:] + xi) + src_y
    xx = x.flatten()
    yy = y.flatten()
    xy = np.array([xx,yy])
    NN_counts = NN_counts.flatten()
    f = griddata(xy.T,NN_counts,(ra1,dec1),method='linear')
    f = f.reshape(100,100)
    psf = 1 / (2*np.pi*sky_psf**2) *np.exp(-(ra1**2 + dec1**2)/2/sky_psf**2)
    psf1 = psf.reshape(100,100)
    convolved_map = signal.convolve(f, psf1,mode='same') / np.sum(psf)
    return convolved_map.ravel()

def log_likelihood(par):
    r2 = (ra1 - par[0])**2 + (dec1 - par[1])**2
    crab = par[2] / (2*np.pi*sky_psf**2) * np.exp(-r2 / 2 / sky_psf**2)
    J0543 = 1e13*anistropy_analytical(par[3:-1])
    diffuse_norm = par[-1]
    model = crab + J0543 + 10**diffuse_norm*diffuse_bkg + croff
#    np.savetxt("aa_best_fit.txt", model)
#    likelihood = -0.5*np.sum((con-model)**2)
    likelihood = 0
    for i in range(10000):
        likelihood += lnpoisson(model[i], con[i])
    w = list(par)
    w.append(likelihood)
    pid = os.getpid()
    f = open(str(pid) + '.csv', 'a')
    csv_writer = csv.writer(f, dialect='excel')
    csv_writer.writerow(w)
    f.close()
    return likelihood
    
def log_prior(par):
    if np.all(par>pl) and np.all(par<pu):         
        return 0.0
    return -np.inf

def log_probability(par):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(par)


################# read LHAASO Geminga_data grid
filename1 = "J0543_25-100_R5"
hdul = fits.open(filename1 + ".fits")
data = np.array(hdul[1].data)
hdul.close()
ra0 = data["RA"]
dec0 = data["DEC"]
con = data["Events number"]
croff = data["Reduced Background events"]
coff = data["Background events"]
diffuse_bkg = 10**data["Diffuse Background"]
sky_ra, sky_dec, sky_psf = 85.79025, 23.4847222, 0.295
ra1, dec1 = coordinate_transform(ra0, dec0, sky_ra, sky_dec)

#################  crab      ###################
#ra_crab, dec_crab = 83.633, 22.014
#ra1_crab, dec1_crab = coordinate_transform(np.array([ra_crab]), np.array([dec_crab]), sky_ra, sky_dec)
#r2 = (ra1 - ra1_crab)**2 + (dec1 -dec1_crab)**2
#print(ra1_crab, dec1_crab)
############### run MCMC ####################

with Pool(nwalkers) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,ndim,log_probability,args=(),pool=pool
    )
    sampler.run_mcmc(par, nstep,progress=True)

#flat_samples = sampler.get_chain(discard=nburn, flat=True)
#df = pd.DataFrame(flat_samples)
#df.to_csv(r'para.csv')
#print(con)

#pars = np.array([-2.016, -1.4299, 33.4396, 0.18, -0.755, -1.525, 1.12,28.047, 0.26, 9.33475, -22.72])
#pars = np.array([-2.01809088e+00,-1.43611277e+00,3.27720106e+01,-1.59413396e-01,7.05598138e-01,-3.10052610e-01,-1.09441168e+00,5.03749914e-01,1.57054100e+00,2.67160724e+01,2.78702281e+01,4.96480157e-01,9.12891626e+00,-2.27223341e+01])
#a=log_likelihood(pars)
#a=anistropy_numerical(pars[3:-1])
#print(a)
