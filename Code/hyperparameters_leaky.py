#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:39:17 2019

@author: adityapandey
"""


import math
import pandas as pd
import numpy as np
min_error = math.inf
min_k = -1
min_n = -1

bound = 3


errors = []

for k in np.arange(0.5,2.0,0.01):
    for n in np.arange(1.1,2.0,0.1):
        
        if(k <= math.pow(bound, 1-n)):
            continue

        flag=False
        error = 0
        rel_error = 0
        abs_error = 0
        for x in np.arange(-bound,bound,0.005):
            if x<0:
                a_relu = -0.01*k*math.pow(abs(x), n)
                actual = 0.01*x
            else:
                a_relu = k*math.pow(x, n)
                actual = x
            
            sq_err = math.pow((a_relu - actual), 2)
            error += sq_err
            abs_err = abs(a_relu - actual) 
            rel_error += abs_err / abs(actual)
            abs_error += abs_err
            
            if abs(k*math.pow(abs(x), n-1)) > 1.1:
                flag=True
                break

        if(flag):
            continue
        errors.append([k, n, error, rel_error, abs_error])
        
error_df = pd.DataFrame(errors, columns = ['k', 'n', 'MSE', 'Rel Error', 'Abs Error']) 

rel_error = 0
rels = []
for x in np.arange(1,bound,1):
            a_relu = 0.54*math.pow(x, 1.3)
            sq_err = math.pow(a_relu - x, 2)
            error += sq_err
            abs_err = math.pow(sq_err, 0.5) 
            a_relu_n_1 = 1.3*0.54*math.pow(x, 0.3)
            rel_error += math.fabs(a_relu - x)/ x
            rels.append([x,a_relu, a_relu_n_1, rel_error])
            
        #print(k,n)
        
'''
For range -3 to 3 (99.7% std normal)
    k = 0.94 n = 1.1
    k = 0.86 n = 1.2
    k = 0.79 n = 1.3
    k = 0.70 n = 1.4
    
For range -1 to 1
    k = 1.05 n = 1.1
    k = 1.07 n = 1.2
    k = 1.09 n = 1.3
    k = 1.09 n = 1.4
    
    
For range -10 to 10
    k = 0.84 n = 1.1
    k = 0.68 n = 1.2
    k = 0.55 n = 1.3
'''
        

'''

Leaky A-ReLU - 
    - alpha = 0.01
    - f(x) - | kxˆn when x>0 
             | 0 when x=0
             | -0.01k|x|ˆn when x<0

    - 

'''

'''

k = math.pow(t, 1-n)*(2*n+1)/(n+2)


k * math.pow(t, 2*n+1)*math.log(2*n+1)/(2*n+1) - 
􏰄 t2n+1 ln(2n + 1) t2n+1 􏰅 tn+2 ln(n + 2) tn+2
k 2n + 1 − (2n + 1)2 = n + 2 − (n + 2)2

'''