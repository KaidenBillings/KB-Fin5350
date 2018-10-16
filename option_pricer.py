#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:10:20 2018

@author: kaidenbillings
"""

import numpy as np
from scipy.stats import binom

S = 80
K = 95
r = .08
v = .3 ### standard deviation
q = 0 ### dividend payouts
expiry = 1 ### T in notes
n = 1

def call_payoff(spot, strike):
    return np.maximum(spot - strike, 0.0)

def put_payoff(spot, strike):
    return np.maximum(strike - spot, 0.0)

### risk neutral form for single period
def single_period_binom_model(S, K, r, v, T, n, payoff):
    h = T/n
    u = np.exp((r-q)*h+v*np.sqrt(h))
    d = np.exp((r-q)*h-v*np.sqrt(h))
    fu = payoff(u*S, K)
    fd = payoff(d*S, K)
    pstar = (np.exp((r - q)*h) - d)/(u - d)
    f0 = np.exp(-r*h)*(fu*pstar + fd*(1-pstar))
    
    return f0

call_price_rn1 = single_period_binom_model(S, K, r, v, expiry, n, call_payoff)

put_price_rn1 = single_period_binom_model(S, K, r, v, expiry, n, put_payoff)

### No-Arbitrage form
def no_arbitrage_form(S, K, r, v, T, n, payoff):
    h = T/n
    u = 1.3
    d = .8
    fu = payoff(u*S, K) ###Cu
    fd = payoff(d*S, K) ###Cd
    D = (fu-fd)/(S*(u-d)) ### Delta
    B = np.exp(-r*h)*((u*fd - d*fu)/(u-d)) ###leverage position
    f_no_arb = D*S + B
    
    return "Premium: " + str(f_no_arb) + " Delta: " + str(D) + " Leverage Position: " + str(B)

call_price_na1 = no_arbitrage_form(S, K, r, v, expiry, n, call_payoff)

put_price_na1 = no_arbitrage_form(S, K, r, v, expiry, n, put_payoff)

### multi period binomial pricer
def euro_binomial_pricer(S, K, r, v, q, T, n, payoff, verbose = True):
    nodes = n  + 1
    h = T / n
    u = np.exp((r - q) * h + v * np.sqrt(h))
    d = np.exp((r - q) * h - v * np.sqrt(h))
    pstar = (np.exp((r - q) * h) - d) / (u - d)
    
    price = 0.0
    
    for i in range(nodes):
        prob = binom.pmf(i, n, pstar)
        spotT = S * (u ** i) * (d ** (n - i))
        po = payoff(spotT, K) 
        price += po * prob
        if verbose:
            print(f"({spotT:0.4f}, {po:0.4f}, {prob:0.4f})")
        
    price *= np.exp(-r * T)
    
    return price


### Problem 1a
S = 100
K = 105
r = .08
expiry = .5
q = 0
n = 1
call_price_na1 = no_arbitrage_form(S, K, r, v, expiry, n, call_payoff)
print("Problem 1a: \n" + call_price_na1)


### 1b
put_price_na1 = no_arbitrage_form(S, K, r, v, expiry, n, put_payoff)
print("\nProblem 1b: \n" + put_price_na1)


### Problem 2
S = 100
K = 95
r = .08
expiry = .5
q = 0
n = 1
### 2a
put_price_na1 = no_arbitrage_form(S, K, r, v, expiry, n, put_payoff)
print("\nProblem 2a: \n" + put_price_na1)
### 2b
print("\nProblem 2b:\nBuy .3 shares and borrow $37.47")
### 2c
print("\nProblem 2c:\nShort .3 shares and invest $37.47 in T-bills")


### Problem 3
S = 100
K = 95
v = .3
r = .08
expiry = 1
q = 0
n = 2
print("\nProblem 3:\n")
euro_binomial_pricer(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
print("\nPrice: " + str(euro_binomial_pricer(S, K, r, v, q, expiry, n, call_payoff, verbose = False)))


### Problem 4
n = 3













