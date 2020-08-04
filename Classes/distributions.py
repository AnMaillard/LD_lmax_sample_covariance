#Some classical probability distributions
import numpy as np
import scipy.special as sc

#The density of the semicircle law in (-2,2)
def sc_density(x):
    return np.sqrt(4.-x**2)/(2*np.pi)

def cdf_sc(y):
    if y <= -2:
        return 0.
    elif y >= 2:
        return 1.
    else:
        return (1./2) + (y*np.sqrt(4.-y**2) + 4*np.arcsin(y/2.))/(4*np.pi)

#The density of the marchenko-pastur law in with ratio gamma >= 1
#We take the convention : MP is the asymptotic spectral law of (1/m) \sum_{\mu =1}^m z_\mu z_\mu^T, with m/n = gamma
def lambdamax(gamma):
    return (1+np.sqrt(1./gamma))**2

def lambdamin(gamma):
    return (1-np.sqrt(1./gamma))**2

def mp_density(x,gamma):
    return gamma*np.sqrt((x-lambdamin(gamma))*(lambdamax(gamma)-x))/(2*np.pi*x)

def cdf_mpastur(x,gamma):
    #The CDF of the Marchenko-Pastur law with parameter gamma >= 1
    if gamma > 1:
        if x <= lambdamin(gamma):
            return 0
        elif x >= lambdamax(gamma):
            return 1
        else:
            squareroot = np.sqrt((x-lambdamin(gamma))*(lambdamax(gamma)-x))
            return 1./2 + gamma*squareroot/(2*np.pi)-(1+gamma)*np.arctan((1+gamma-x*gamma)/(gamma*squareroot))/(2*np.pi)-(gamma-1)*np.arctan((-1.+ gamma* (2+x+gamma*(x-1)) )/ ((gamma-1) * gamma * squareroot))/(2*np.pi)
    elif gamma == 1:
        #The case gamma = 1
        if x <= lambdamin(gamma):
            return 0
        elif x >= lambdamax(gamma):
            return 1. 
        else:
            return 1./2 + (np.sqrt(x*(4-x))/(2*np.pi)) + (1/np.pi)*np.arctan((x-2)/np.sqrt(x*(4-x)))
    else:
        assert False, "ERROR : Non-implemented value of gamma (gamma < 1 ?)"
    
#For the distribution Beta(beta,gamma), with gamma > 1
def cdf_beta(x,beta,gamma):
    #CDF of the beta distribution
    if x <= 0:
        return 0.
    elif x >= 1:
        return 1.
    else:
        return sc.betainc(beta, gamma, x)

def pdf_beta(x,beta,gamma):
    return x**(beta-1.)*(1-x)**(gamma-1.) / sc.beta(beta,gamma)

def cdf_truncated_gaussian(x, delta):
    #The cdf of a truncated Gaussian distribution centered in 1 and with standard deviation delta
    if x <= 0:
        return 0.
    else:
        numerator = sc.erf(1./np.sqrt(2*delta**2)) + sc.erf((x-1.)/np.sqrt(2*delta**2))
        denominator = (1+sc.erf(1./np.sqrt(2*delta**2)))
        return numerator / denominator

def pdf_truncated_gaussian(x,delta):
    if x <= 0:
        return 0.
    else:
        denominator = (1+sc.erf(1./np.sqrt(2*delta**2)))/2.
        numerator = np.exp(-(x-1.)**2/(2.*delta**2))/np.sqrt(2*np.pi*delta**2)
        return numerator / denominator
