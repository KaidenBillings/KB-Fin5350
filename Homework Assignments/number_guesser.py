#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:56:17 2018

@author: kaidenbillings
"""

def print_header():
    print("Think of a number between 1 and 100")
    print("I'm going to try to guess your number")
    print("Tell me if my guess is correct by entering yes, lower or higher")
    
### Write the code for the ending message
def print_footer(guess, tries):
    print("I got it! Your number was %s." % guess)
    print("And it only took me %s tries!" % tries)



def main():
    ###print the greeting banner
    print_header()
    
    ###set the initial values
    max_number = 100
    min_number = 1
    answer = "Take a guess"
    tries = 1
    
    ###write code for guesses
    while answer != "yes":
        guess = (max_number + min_number)//2
        print("\nMy guess is %s" % guess)
        answer = str(input("Was I right? "))
        if answer == "higher":
            min_number = guess + 1
            tries += 1
        elif answer == "lower":
            max_number = guess - 1
            tries += 1
    print_footer(guess, tries)
    
main()