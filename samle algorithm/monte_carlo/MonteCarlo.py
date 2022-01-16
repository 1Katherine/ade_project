#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MonteCarlo.py   
@Author ：Yang 
@CreateTime :   2022/1/15 22:50 
@Reference : https://towardsdatascience.com/monte-carlo-simulation-and-variants-with-python-43e3e7c59e1f
'''
import numpy as np
import statsmodels.api as sm

np.random.seed(2021)
mu = 0
sigma = 1
n = 100
# assumed population parameters
alpha = np.repeat(0.5, n)
beta = 1.5


def MC_estimation_slope(M):
    MC_betas = []
    MC_samples = {}

    for i in range(M):
        # randomly sampling from normal distribution as error terms
        e = np.random.normal(mu, sigma, n)
        # generating independent variable by making sure the variance in X is larger than the variance in error terms
        X = 9 * np.random.normal(mu, sigma, n)
        # population distribution using the assumd parameter values alpha/beta
        Y = (alpha + beta * X + e)

        # running OLS regression for getting slope parameters
        model = sm.OLS(Y.reshape((-1, 1)), X.reshape((-1, 1)))
        ols_result = model.fit()
        coeff = ols_result.params

        MC_samples[i] = Y
        MC_betas.append(coeff)
    MC_beta_hats = np.array(MC_betas).flatten()
    return (MC_samples, MC_beta_hats)


MC_samples, MC_beta_hats = MC_estimation_slope(M=10000)
beta_hat_MC = np.mean(MC_beta_hats)

print(MC_beta_hats)

print(beta_hat_MC)

