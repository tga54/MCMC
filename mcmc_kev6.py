#!/bin/bash
import re
import os
import csv
import secrets
from scipy import interpolate
import time
import numpy as np
import pandas as pd
import sys
sys.path.append(r'/opt/anaconda3/lib/python3.6/site-packages/emcee-0.0.0-py3.6.egg/')
import emcee
from multiprocessing import Pool
t1 = time.time()
erg = 6.63e-27*np.logspace(17.2,18.3,12)
log_erg = np.log(erg)
mean_log_erg = np.mean(log_erg)
erg_max = 7000*1.6e-12
erg_min = 700*1.6e-12
def pwn_model(eta,s,B0,D0,delta,gmax):
    name = secrets.token_hex(8)
    code=open(r'kev.f90','r')
    script = code.readlines()
    code.close()
    script[27] = 'filename=\'' + name + '\'\n'
    script[28] = 'eta=10**' + str(eta)[:8] + '\n'
    script[29] = 's=' + str(s)[:8] + 'd0\n'
    script[30] = 'B0=10**' + str(B0)[:8] + '\n'
    script[31] = 'D0=10**' + str(D0)[:8] + '\n'
    script[32] = 'delta=' + str(delta)[:8] + 'd0\n'
    script[34] = 'gmax=10**' + str(gmax)[:8] + '\n'
    code = open(name+'.f90','w')
    for s in script:
        code.write(s)
    code.close()
    os.system(r'/opt/intel/oneapi/compiler/2021.2.0/linux/bin/intel64/ifort -qopenmp -o '+ name +' '+name+'.f90')
    os.system(r'./'+name)
    os.system(r'rm '+name)
    os.system(r'rm '+name+'.f90')
    data = open(r'intensity_syn_'+name+'.dat','r')
    data1 = data.readlines()
    data.close()
    os.system(r'rm intensity_syn_'+name+'.dat')
    da = []
    for d in data1:
        a = d.split()
        for n in a:
            da.append(float(n))
    syn = np.array(da)
    syn = syn.reshape(7,12)
    syn = syn/erg[None,:]/(3600*180/3.1416)**2
    log_syn = np.log(syn)
    mean_log_syn = np.mean(log_syn,axis=1)
    b = np.sum((log_erg-mean_log_erg)[np.newaxis,:]* \
               (log_syn-mean_log_syn[:,np.newaxis]),axis=1)/ \
        np.sum((log_erg - mean_log_erg)**2)
    a = mean_log_syn - b*mean_log_erg
    itst = np.exp(a) / (b+1) *(np.power(erg_max,b+1)-np.power(erg_min,b+1))
    index = 1 - b
    result = np.concatenate([itst,index])
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i]=0
    return result
def log_likelihood(par, y, yerr):
    eta, s, B0, D0, delta, gmax = par
    model = pwn_model(eta,s,B0,D0,delta,gmax)
    sigma2 = yerr ** 2 
    likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 )
    w = [eta, s, B0, D0, delta,gmax, likelihood]
    pid = os.getpid()
    f = open(str(pid)+'.csv','a')
    csv_writer = csv.writer(f,dialect='excel')
    csv_writer.writerow(w)
    f.close()
    return likelihood

def log_prior(par):
    eta,s,B0,D0,delta,gmax = par
    if -3<eta<0 and 1.01<s<3.5 and -6<B0<-2 and 25<D0<30 and 0.001<delta<0.99 and 6<gmax<10:
        return 0.0
    return -np.inf
def log_probability(par, y, yerr):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(par, y, yerr)
chan_y = np.array([1.61e-17,9.89e-18,5.27e-18,3.72e-18,3.19e-18,2.46e-18,2.11e-18,1.77,1.84,1.99,2.09,2.22,2.35,2.59])
chan_yerr = np.array([1.22e-18,5.63e-19,2.96e-18,6.41e-19,4.41e-19,3.58e-19,3.03e-19,0.07,0.06,0.08,0.09,0.1,0.12,0.14])
pos = np.array([-1.8,1.7,-4.87,27.88,0.01,8.37])*(1 + 0.01*np.random.randn(12,6))
nstep = 5000
nwalkers, ndim = pos.shape
with Pool(nwalkers) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(chan_y, chan_yerr),pool=pool
        )
    sampler.run_mcmc(pos, nstep,progress=True)
flat_samples = sampler.get_chain(discard=0,flat=True)
df = pd.DataFrame(flat_samples)
df.to_csv(r'para_ndiv10.csv')
t2 = time.time()
print(t2-t1)
