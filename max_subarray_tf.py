# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:47:59 2019
@author: mritch
Defines functions for both 1D and 2D max subarray finding procedures operating on tensors
"""

import tensorflow as tf


def kadane_tf(vec):
    """
    Implements Kadane's algorithm for 1D tensor
    Input: 1D tensor
    Outputs: sum of max subarray, start and finish indices for max subarray
    (max_all,imax1,imax2)=kadane_tf(vec)
    """
    #initialize
    max_all=tf.math.abs(vec[0])*-2
    max_here=tf.math.abs(vec[0])*-2
    imax1=0
    imax2=0
    ihere1=0
    ihere2=0
    #make tensor vector iterable
    vec=tf.unstack(vec)
    for i, curr in enumerate(vec):
        #possibility 1: this value is greatest
        opt1=curr
        #possibility 2: this value + previous max is greatest
        opt2=(curr)+max_here
        #check which of these two is greatest
        max_here=tf.cond(opt1<opt2, lambda: opt2,lambda: opt1)
        (ihere1,ihere2)=tf.cond(opt1<opt2, lambda: (ihere1,i),lambda: (i,i))
        #check if this is greater than global max
        (imax1,imax2)=tf.cond(max_all<max_here, lambda: (ihere1,ihere2),lambda: (imax1,imax2))
        max_all=tf.cond(max_all<max_here, lambda: max_here,lambda: max_all)
        
    return max_all,imax1,imax2



def maxSubArray_2D(mat):
    """
    Implements max subarray procedure for 2D tensors
    Input: 2D tensor
    Outputs: sum of max subarray, left, right, top, and bottom indices for max subarray rectangle
    (maxRecSum,maxRecL,maxRecR,maxRecT,maxRecB)=maxSubArray_2D(mat)
    """
    #initialize
    shape=tf.shape(mat)
    maxRecSum=tf.math.abs(mat[0,0])*-2
    maxRecL=0
    maxRecR=0
    maxRecT=0
    maxRecB=0
    #make iterable list of column tensors
    mat=tf.unstack(mat,axis=1)
    for L in range(len(mat)):
        rem=mat[L:]
        rowsums=tf.zeros(shape[0],1)
        for R, curr in enumerate(rem):
            #keep running tally of rowsums
            rowsums=rowsums+curr
            #do kadane on rowsum
            (max_here,T,B)=kadane_tf(rowsums)
            #check if is greater than global max
            (maxRecL,maxRecR,maxRecT,maxRecB)=tf.cond(maxRecSum<max_here, lambda: (L,R+L,T,B),lambda: (maxRecL,maxRecR,maxRecT,maxRecB)) #return indices
            maxRecSum=tf.cond(maxRecSum<max_here, lambda: max_here,lambda: maxRecSum) #return max
        
            
    
    return maxRecL,maxRecR,maxRecT,maxRecB,maxRecSum
    
