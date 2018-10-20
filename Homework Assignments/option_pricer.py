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

### multi period recursive binomial pricer
def euro_binomial_pricer_recursive(S, K, r, v, q, T, n, payoff, verbose = True):
    nodes = n  + 1
    h = T / n
    u = np.exp((r - q) * h + v * np.sqrt(h))
    d = np.exp((r - q) * h - v * np.sqrt(h))
    pu = (np.exp((r - q) * h) - d) / (u - d)
    pd = 1.0 - pu
    disc = np.exp(-r * h)
    
    
    ## Arrays to store the spot prices and option values
    Ct = np.empty(nodes)
    St = np.empty(nodes)
    
    for i in range(nodes):
        St[i] = S * (u ** (n - i)) * (d ** i)
        Ct[i] = payoff(St[i], K)
    
    if verbose:
        print(Ct)
        
    for t in range((n - 1), -1, -1):
        for j in range(t+1):
            Ct[j] = disc * (pu * Ct[j] + pd * Ct[j+1])
            # St[j] = St[j] / u
            # Ct[j] = np.maximum(Ct[j], early payoff)
            print(Ct)
            
    return Ct[0]

### mutli period recursive model that shows full price tree
def euro_binom_pricer_recursive_matrix(S, K, r, v, q, T, n, payoff, verbose = True):
    nodes = n  + 1
    h = T / n
    u = np.exp((r - q) * h + v * np.sqrt(h))
    d = np.exp((r - q) * h - v * np.sqrt(h))
    pu = (np.exp((r - q) * h) - d) / (u - d)
    pd = 1.0 - pu
    disc = np.exp(-r * h)
    
    ## Arrays to store the spot prices and option values
    Ct = np.zeros((nodes, n+1))
    St = np.zeros((nodes, n+1))
    Dt = np.zeros((nodes, n+1))
    Bt = np.zeros((nodes, n+1))
    
    ### Loop to calculate terminal values
    for i in range(nodes):
        St[i, n] = S * (u**(n-i)) * (d**i)
        Ct[i, n] = payoff(St[i, n], K)
    
    ### Recursive loop to calculate prior values
    for t in range((n-1), -1, -1):  # for t in range ((start at n-1), go up to but not including -1, 
                                    # and count backward by 1)
        for j in range(t+1): #When t is at column 1, j will reference rows 0 up2ni 2 (t+1), so 0 - 1
            St[j, t] = St[j, t+1] / u
            Ct[j, t] = disc * ((pu * Ct[j, t+1]) + (pd * Ct[j+1, t+1]))
            Dt[j, t] = (Ct[j, t+1] - Ct[j+1, t+1])/(St[j, t]*(u-d))
            Bt[j, t] = disc * ((u*Ct[j+1, t+1] - d*Ct[j, t+1])/(u-d))
        
    if verbose:
        print("\nStock Tree")
        print(St)
        print("\nPremium Tree")
        print(Ct)
        print("\nDelta Tree")
        print(Dt)
        print("\nLeverage Tree")
        print(Bt)
        print("\n")           
            
    return print(f"Premium: {Ct[0,0]:0.4f}")


### multi period american binomial pricer with recursive matrix
def american_binom_pricer_recursive_matrix(S, K, r, v, q, T, n, payoff, verbose = True):
    nodes = n  + 1
    h = T / n
    u = np.exp((r - q) * h + v * np.sqrt(h))
    d = np.exp((r - q) * h - v * np.sqrt(h))
    pu = (np.exp((r - q) * h) - d) / (u - d)
    pd = 1.0 - pu
    disc = np.exp(-r * h)
    
    ## Arrays to store the spot prices and option values
    Ct = np.zeros((nodes, n+1))
    St = np.zeros((nodes, n+1))
    
    ### Loop to calculate terminal values
    for i in range(nodes):
        St[i, n] = S * (u**(n-i)) * (d**i)
        Ct[i, n] = payoff(St[i, n], K)
    
    ### Recursive loop to calculate prior values
    for t in range((n-1), -1, -1):   
        for j in range(t+1): 
            St[j, t] = St[j, t+1] / u
            Ct[j, t] = disc * ((pu * Ct[j, t+1]) + (pd * Ct[j+1, t+1]))
            Ct[j, t] = np.maximum(Ct[j, t], payoff(St[j, t], K))
          
    if verbose:
        print("\nStock Tree")
        print(St)
        print("\nPremium Tree")
        print(Ct)
            
    return print(f"Premium: {Ct[0,0]:0.4f}")

### Just the stock tree
def binomial_stock_tree(S, r, v, q, T, n):
    nodes = n  + 1
    h = T / n
    u = np.exp((r - q) * h + v * np.sqrt(h))
    d = np.exp((r - q) * h - v * np.sqrt(h))
    St = np.zeros((nodes, n+1))
    for i in range(nodes):
        St[i, n] = S * (u**(n-i)) * (d**i)
    for t in range((n-1), -1, -1):   
        for j in range(t+1): 
            St[j, t] = St[j, t+1] / u
    return St, u, d

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
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)


### Problem 4
n = 3
S = 80
print("\nProblem 4:\n")
print("Stock price: " + str(S))
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
S = 90
print("Stock price: " + str(S))
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
S = 110
print("Stock price: " + str(S))
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
S = 120
print("Stock price: " + str(S))
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
S = 130
print("Stock price: " + str(S))
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)

### Problem 5
S = 100
K = 95
r = .08
v = .3
q = 0
expiry = 1
n = 3
print("\nProblem 5:")
print("\n5a:")
american_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = True)
print("There is no early exercise.\n")
print("5b:")
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = False)
print("The premiums are the same.\n")
print("5c:")
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, put_payoff, verbose  = False)
print("\nDoes parity hold?")
print("Synthetic Stock:")
print(euro_binomial_pricer(S, K, r, v, q, expiry, n, call_payoff, verbose = False) - \
euro_binomial_pricer(S, K, r, v, q, expiry, n, put_payoff, verbose = False) + np.exp(-r*expiry)*K)
print("Stock Price:")
print(S)
print("Yes. absent rounding errors, parity holds")
print("\n5d:")
american_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, put_payoff, verbose = True)
print("The american put premium is higher because the option to exercise early is exercised.")

### Problem 6
S = 40
K = 40
r = .08
v = .3
q = 0
expiry = .5
n = 3

print("\nProblem 6:")
print("6a:")
print(binomial_stock_tree(S, r, v, q, expiry, n))
print("\n6b:")
print("American Call:")
american_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = False)
print("American Put:")
american_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, put_payoff, verbose = False)
print("European Call:")
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, call_payoff, verbose = False)
print("European Put:")
euro_binom_pricer_recursive_matrix(S, K, r, v, q, expiry, n, put_payoff, verbose = False)


















