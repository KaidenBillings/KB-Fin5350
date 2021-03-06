#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:06:02 2018

@author: kaidenbillings
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm



### BSM formulas
def bsm_call(spot, strike, div, expiry, rate, vol):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*expiry)/(vol*np.sqrt(expiry))
    d2 = d1 - vol*np.sqrt(expiry)
    call_price = spot*np.exp(-div*expiry)*norm.cdf(d1) - strike*np.exp(-rate*expiry)*norm.cdf(d2)
    return call_price

def bsm_put(spot, strike, div, expiry, rate, vol):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*expiry)/(vol*np.sqrt(expiry))
    d2 = d1 - vol*np.sqrt(expiry)
    put_price = strike*np.exp(-rate*expiry)*norm.cdf(d2) - spot*np.exp(-div*expiry)*norm.cdf(d1)
    return put_price

### option greeks
### Delta
def bsm_call_delta(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    delta = np.exp(-div*tau)*norm.cdf(d1)
    return delta

def bsm_put_delta(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    delta = np.exp(-div*tau)*norm.cdf(-d1)
    return delta

### Gamma
def bsm_gamma(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    gamma = (np.exp(-div*tau)*norm.pdf(d1))/(spot*vol*np.sqrt(tau))
    return gamma
    
### Theta
def bsm_call_theta(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    d2 = d1 - vol*np.sqrt(tau)
    theta = div*spot**(-div*tau)*norm.cdf(d1) - rate*strike*np.exp(-rate*tau)*norm.cdf(d2) - \
    (strike*np.exp(-rate*tau)*norm.pdf(d2)*vol)/(2*np.sqrt(tau))
    return theta

def bsm_put_theta(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    d2 = d1 - vol*np.sqrt(tau)
    theta = div*spot**(-div*tau)*norm.cdf(d1) - rate*strike*np.exp(-rate*tau)*norm.cdf(d2) - \
    (strike*np.exp(-rate*tau)*norm.cdf(d2)*vol)/(2*np.sqrt(tau)) + rate*strike*np.exp(-rate*tau) - div*spot*np.exp(-div*tau)
    return theta

### Vega
def bsm_vega(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    vega = spot*np.exp(-div*tau)*norm.pdf(d1)*np.sqrt(tau)
    return vega

### Rho
def bsm_call_rho(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    d2 = d1 - vol*np.sqrt(tau)
    rho = tau*strike*np.exp(-rate*tau)*norm.cdf(d2)
    return rho

def bsm_put_rho(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    d2 = d1 - vol*np.sqrt(tau)
    rho = -tau*strike*np.exp(-rate*tau)*norm.cdf(-d2)
    return rho

### Psi
def bsm_call_psi(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    psi = -tau*spot*np.exp(-div*tau)*norm.cdf(d1)
    return psi

def bsm_put_psi(spot, strike, vol, rate, tau, div):
    d1 = (np.log(spot/strike)+(rate-div+.5*vol*vol)*tau)/(vol*np.sqrt(tau))
    psi = tau*spot*np.exp(-div*tau)*norm.cdf(-d1)
    return psi

def plot_price_path(path):
    nsteps = path.shape[0]
    plt.plot(path, 'b', linewidth = 2.5)
    plt.title("Simulated Binomial Price Path")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price ($)")
    plt.xlim((0, nsteps))
    plt.grid(True)
    plt.show()


### problem 1
K = 40
r = .08
v = .3
T = 1
q = 0
S = 80
x = np.arange(80)

### 1b
call_delta_plot = np.zeros(S)
call_gamma_plot = np.zeros(S)
call_vega_plot = np.zeros(S)
call_theta_plot = np.zeros(S)
call_rho_plot = np.zeros(S)
for S in range(1,S): ### for some reason there is a division by zero when the range starts at zero
    call_delta_plot[S] = bsm_call_delta(S, K, v, r, T, q)
    call_gamma_plot[S] = bsm_gamma(S, K, v, r, T, q)
    call_vega_plot[S] = bsm_vega(S, K, v, r, T, q)
    call_theta_plot[S] = bsm_call_theta(S, K, v, r, T, q)
    call_rho_plot[S] = bsm_call_rho(S, K, v, r, T, q)
plt.plot(x, call_delta_plot)
plt.plot(x, call_gamma_plot)
plt.plot(x, call_vega_plot)
plt.plot(x, call_theta_plot)
plt.plot(x, call_rho_plot)
    
### 1c
put_delta_plot = np.zeros(S)
put_gamma_plot = np.zeros(S)
put_vega_plot = np.zeros(S)
put_theta_plot = np.zeros(S)
put_rho_plot = np.zeros(S)
for S in range(1,S):
    put_delta_plot[S] = bsm_call_delta(S, K, v, r, T, q)
    put_gamma_plot[S] = bsm_gamma(S, K, v, r, T, q)
    put_vega_plot[S] = bsm_vega(S, K, v, r, T, q)
    put_theta_plot[S] = bsm_call_theta(S, K, v, r, T, q)
    put_rho_plot[S] = bsm_call_rho(S, K, v, r, T, q)
plt.plot(x, put_delta_plot)
plt.plot(x, put_gamma_plot)
plt.plot(x, put_vega_plot)
plt.plot(x, put_theta_plot)
plt.plot(x, put_rho_plot)


### problem 2
S = 41
n = 1000
T = 5
h = T/n

price_path = np.zeros(n)
price_path[0] = S
for t in range (1,n):
    z = np.random.normal(size = 1)
    price_path[t] = price_path[t-1]*np.exp((r - q - .5*v**2)*h + v*np.sqrt(h)*z)

plot_price_path(price_path)
    
    


    
    
    
    
    
    
    
    
    
    
    
    
plt.show()