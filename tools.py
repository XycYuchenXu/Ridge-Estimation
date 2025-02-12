#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xycxu
"""

#%% ridge def
import numpy as np
import math

from itertools import combinations_with_replacement
from scipy import ndimage as ndi
from scipy.signal import convolve2d

def _divide_nonzero(array1, array2, cval=1e-9):
    """
    Divides two arrays.

    Denominator is set to small value where zero to avoid ZeroDivisionError and
    return finite float array.

    Parameters
    ----------
    array1 : (N, ..., M) ndarray
        Array 1 in the enumerator.
    array2 : (N, ..., M) ndarray
        Array 2 in the denominator.
    cval : float, optional
        Value used to replace zero entries in the denominator.

    Returns
    -------
    array : (N, ..., M) ndarray
        Quotient of the array division.
    """

    # Copy denominator
    denominator = np.copy(array2)

    # Set zero entries of denominator to small value
    denominator[denominator == 0] = cval
    denominator[abs(denominator) < cval] = cval * np.sign(denominator[abs(denominator) < cval])

    # Return quotient
    return np.divide(array1, denominator)

#%% apply noise
def noise_apply(image, mu, sd, type = 'p'):
    """
    Apply the noise with specified signal to noise level and distribution type.

    Parameters
    ----------
    image : (M, N, T) ndarray
        The clean video (without noise) to apply noise to, which must have its reference level of pixel values identical to 1.
    mu : float
        The targeted reference mean pixel values of the noisy video after the step.
    sd : float
        The targeted reference standard deviation of the noisy video after the step. Both `sd` and `mu` can be the statistics for a sub-region of the video array.
    type : str, optional
        Whether to apply additive Gaussian noise ('g') or poisson type noise ('p'), by default 'p'.

    Returns
    -------
    (M, N, T) ndarray
        The video with noise applied.
    """
    if type == 'p':
        a = sd**2/mu - 1
        image_noisy = np.random.poisson(np.random.poisson(image / a) * a)
    elif type == 'g':
        image_noisy = image * mu + np.random.normal(0, sd, image.shape)
    return image_noisy

    
#%% ridge estimation

def ridge_merge(image, sigmas, deltas, edge, mode = 'nearest', cval = 0):
    """
    The main function of estimating a ridge (in a callable format) from the supplied video.

    Parameters
    ----------
    image : (M, N, T) ndarray
        Video sequence of length T that supposedly contains one ridge.
    sigmas : list of int
        Candidate (positive) widths for the spatial filter.
    deltas : list of int
        Candidate (positive) widths for the temporal filter.
    edge : int
        Further shrink the interested area by cropping out edge pixels at all 4 directions before applying the filter.
    mode : str, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border, with the same choices as the function `gaussian_filter` in `scipy.ndimage`, by default (and recommended) 'nearest'.
    cval : int, optional
        Value to fill past edges of input if mode is 'constant', by default 0.

    Returns
    -------
    dict
        'curve': callable
            A function that takes time index `t` and kernel width `rho` as the input to output the location of the estimated ridge at time `t`.
        'score': (M-2*edge, N-2*edge, T) ndarray
            The weights array for pseudo-likelihood.
        'cov': (T, 2, 2) ndarray
            The proxy covariance matrices of the location estimators at all `T` time indices.
        'tangent': (T, 3) ndarray
            The tangent of the estimated ridge function at all `T` time indices.
        'vector': (M-2*edge, N-2*edge, T, 3) ndarray
            The normalized pixel wise gradient array.
        'intensity': (T,) 1darray
            The intensity estimates at all `T` time indices.
        'intvar': (T,) 1darray
            The proxy variance of the intensity estimates at all `T` time indices.
    """
    image = image.astype(np.float64)
    M, N, T = image.shape
    filtered = np.zeros((M-2*edge, N-2*edge, T))
    tangents = np.zeros((M-2*edge, N-2*edge, T, 3))
    laplace = np.zeros((M-2*edge, N-2*edge, T, 3))
    
    def score(i):
        L_v = 2 * abs(np.sum(g_n * v[...,i], axis = -1))# * np.log(1 + vl)
        d_i = d[...,i]
        d_sum = np.sum(d, axis=-1) - d_i # lamb1 + lamb2
        d_prod = d_sum ** 2 - np.sum(d ** 2, axis=-1) + d_i ** 2 # 2 * lamb1 * lamb2
        prod_ratio = _divide_nonzero(d_prod, d_sum ** 2 - d_prod)
        L_l = 2 * prod_ratio + 2 * prod_ratio * \
            np.log(1 + np.maximum(d_prod / 2 - d_i ** 2, 0))
    
        L_t = 4 * abs(v[...,-1,i])
        return np.exp(L_t + L_l + L_v)
    
    for sigma in sigmas:
        eta_t = np.zeros_like(filtered)
        tg_t = np.zeros_like(tangents)
        lap_t = np.zeros_like(tangents)
        for delta in deltas:
            var = [sigma, sigma, delta]
            # Make nD hessian
            img = ndi.gaussian_filter(image, sigma=var, truncate=2,
                                      mode=mode, cval=np.mean(image))
            #img = L * (img - np.quantile(img, 0.25)) / (np.quantile(img, 0.75) - np.quantile(img, 0.25))
            img = img[edge:-edge, edge:-edge,:]
    
            gradients = np.gradient(img, edge_order=2)
            axes = range(img.ndim)
    
            hessian_elements = [np.gradient(gradients[ax0], axis=ax1, edge_order=2) * var[ax1] * var[ax0]
                                for ax0, ax1 in combinations_with_replacement(axes, 2)]
            gradients = np.stack(gradients, axis = -1) * np.array(var)[None,None,None,...]
            temp_shape = hessian_elements[0].shape
            hessM = np.zeros(temp_shape + (len(temp_shape), len(temp_shape)))
            for idx, (row, col) in \
                    enumerate(combinations_with_replacement(range(len(temp_shape)), 2)):
                hessM[..., row, col] = hessian_elements[idx]
                hessM[..., col, row] = hessian_elements[idx]
    
            d, v = np.linalg.eigh(hessM)
            v = v * np.where(v[...,-1,None,:] >= 0, 1, -1)
            vl0 = np.sqrt(np.sum(gradients**2, axis = -1))
            LL = np.median(vl0)
    
            g_n = np.sum(gradients[...,None] * v, axis=-2)
            g_n = _divide_nonzero(g_n, d)
            g_n = np.sum(g_n[...,None,:] * v, axis=-1)
            vl = np.sqrt(np.sum(g_n**2, axis = -1))
            g_n = _divide_nonzero(g_n, vl[...,None])
            
            L_star = _divide_nonzero(np.sum(g_n * gradients, axis = -1), vl * LL)
            theta = abs(_divide_nonzero(L_star * LL * vl, vl0))
            
            d /= LL
            g_n = g_n * np.where(g_n[...,-1,None] >= 0, 1, -1)
                
            d_sum = np.sum(d, axis = -1) - L_star
            d_sum2 = np.sum(d ** 2, axis = -1) - L_star ** 2
            d_prod = d_sum ** 2 - d_sum2
            prod_ratio = _divide_nonzero(d_prod, d_sum2)
            
            eta1 = np.exp(4 * theta + 2 * prod_ratio + \
                          #2 * g_n[...,-1] + \
                          2 * prod_ratio * np.log(1 + np.maximum(d_prod / 2 - L_star ** 2, 0)))
    
            eta2 = np.stack([score(i) for i in range(3)], axis = -1)
            temp = eta2.argsort()[...,-1]
            eta2 = np.take_along_axis(eta2, temp[...,None], axis=-1)[...,0]
            tg2 = np.take_along_axis(v,
                                     np.repeat(temp[...,None,None], 3, axis=-2),
                                     axis = -1)[...,0]
            
            eta = eta1 + eta2
            tg = _divide_nonzero(eta1[...,None] * g_n + eta2[...,None] * tg2, eta[...,None])
            eta = _divide_nonzero(eta, np.sum(eta, axis = (0,1))[None,None,:])
    
            eta_t += eta
            tg_t += eta[...,None] * tg
            lap_t += eta[...,None] * gradients
            # lap_t += eta * (hessM[...,0,0] + hessM[...,1,1])# - hessM[...,0,1] * hessM[...,1,0])
            
        tangents = np.where(np.repeat(eta_t[...,None], 3, axis = -1) > np.repeat(filtered[...,None], 3, axis = -1),
                            tg_t, tangents)
        laplace = np.where(np.repeat(eta_t[...,None], 3, axis = -1) > np.repeat(filtered[...,None], 3, axis = -1),
                           _divide_nonzero(lap_t, eta_t[...,None]), laplace)
        filtered = np.where(eta_t > filtered, eta_t, filtered)
    
    XY = np.stack(np.meshgrid(range(M-2*edge), range(N-2*edge), indexing='ij'), axis = -1)
    XY_diff = XY[None,None,...,None,:] - XY[...,None,None,None,:]
    eta = filtered / np.sum(filtered, axis = (0,1))[None,None,:]
    gradients = _divide_nonzero(tangents, tangents[...,-1,None])
    
    grad1 = gradients[...,None,None,:-1,:2]; grad2 = gradients[None,None,...,1:,:2]
    tp = np.exp(
       - np.sqrt( np.sum(
           3 * (grad1 + grad2 - 2 * XY_diff) * (XY_diff - grad1)\
               + (3 * XY_diff - 2 * grad1 - grad2) ** 2, axis = -1
           ))# * np.sum(XY_diff ** 2, axis = -1))# * _divide_nonzero(
               # 1 + np.sum(XY_diff ** 2 + (XY_diff - grad1) * (grad2 - XY_diff)/3 + 2 * (grad1 + grad2 - 2 * XY_diff) ** 2 / 15, axis = -1),
               # 1 + np.sum(XY_diff ** 2, axis = -1)))
        )
    psi_f = np.copy(eta); psi_b = np.copy(eta)
    for t in range(T-1):
        eps_f = np.min(np.max(abs(tp[...,t]), axis=(2,3)))
    
        eps_b = np.min(np.max(abs(tp[...,-t-1]), axis=(0,1)))
    
        psi_f[...,t+1] *= np.sum(_divide_nonzero(tp[...,t], np.sum(tp[...,t], axis = (2,3))[...,None,None],
                                                 cval = eps_f if eps_f != 0 else np.finfo(float).eps) \
                      * psi_f[None,None,...,t], axis = (2,3))
        psi_f[...,t+1] = _divide_nonzero(psi_f[...,t+1], np.sum(psi_f[...,t+1]))
    
        psi_b[...,-t-2] *= np.sum(_divide_nonzero(tp[...,-t-1], np.sum(tp[...,-t-1], axis = (0,1))[None,None,...],
                                                  cval = eps_b if eps_b != 0 else np.finfo(float).eps) \
                                  * psi_b[...,None,None,-t-1], axis = (0,1))
        psi_b[...,-t-2] = _divide_nonzero(psi_b[...,-t-2], np.sum(psi_b[...,-t-2]))
    
    powers = np.sqrt(np.column_stack((np.arange(1, T+1), np.arange(T,0,-1))))
    powers = powers / (powers[:,None,0] + powers[:,None,1])
    beta = np.power(psi_f, powers[None,None,:,0]) * np.power(psi_b, powers[None,None,:,1])
    # beta = np.sqrt(psi_f * psi_b)
    # beta = np.power(psi_f, 1/10 + np.arange(1, T+1)[None,None,:]/(1.25*(T+1))) * np.power(psi_b, 1/10 + np.arange(T, 0, -1)[None,None,:]/(1.25 * (T+1)))
    # beta = beta / np.sum(beta, axis = (0,1))[None,None,:]
    # beta = (psi_f + psi_b) / 2
    
    filtered = beta / np.sum(beta, axis = (0,1))[None,None,:]
    gradients = _divide_nonzero(gradients, np.sqrt(np.sum(gradients ** 2, axis = -1))[...,None])
    tangent = np.sum(filtered[...,None] * gradients, axis = (0,1))
    tangent = _divide_nonzero(tangent, tangent[...,-1,None])
    XYT = np.column_stack((np.sum(filtered[...,None] * XY[...,None,:], axis = (0,1)),
                           np.arange(T)))
    
    XY_minus_bar = XY[...,None,:] - XYT[None,None,...,:2]
    Cov = np.sum(np.matmul(XY_minus_bar[...,None], XY_minus_bar[...,None,:]) * filtered[...,None,None],
                 axis = (0,1))# * _divide_nonzero(1, 1 - np.sum(filtered ** 2, axis = (0,1)))[:,None,None]
    varPi = np.sum(laplace[...,:-1] * XY_minus_bar[...,:], axis=-1)
    intensity = np.array([np.sum(convolve2d(varPi[...,i], filtered[...,i], mode='same')) for i in range(T)])/2
    intvar = np.array([np.sum((convolve2d(varPi[...,i], np.ones_like(filtered[...,i])/2, mode='same') - intensity[i]) ** 2 * filtered[...,i]) for i in range(T)])/2
    # intensity = np.sum(np.sum(laplace[...,:-1] * XY_minus_bar, axis = -1), axis = (0,1))# * (Cov[:,0,0] * Cov[:,1,1] - Cov[:,0,1] ** 2) ** 0.5
    def ridge_curve(t, rho):
        w = np.exp( - (t - np.arange(T))**2 / (2 * rho**2))
        #return (*list(np.sum(filtered[...,None] * (XY[...,None,:] + tangents[...,:2] * (t - np.arange(T))[None,None,:,None]) * w[None,None,:,None], axis = (0,1,2)) /sum(w)), t)
        return np.sum((XYT + tangent * (t - np.arange(T))[:,None]) * w[:,None], axis = 0)/sum(w)
        #return list(np.sum(XYT[:,:2] * w[:,None], axis = 0)/sum(w)) + [t]
    return {'curve': ridge_curve, 'score': filtered, 'cov': Cov,
            'tangent': tangent, 'vector': tangents, 'intensity': intensity, 'intvar': intvar}








