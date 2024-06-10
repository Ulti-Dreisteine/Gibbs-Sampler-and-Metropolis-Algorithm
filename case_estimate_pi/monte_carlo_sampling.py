# -*- coding: utf-8 -*-
"""
Created on 2024/06/10 14:24:18

@File -> monte_carlo.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 蒙特卡洛方法求解圆周率
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def estimate_pi(N: int) -> float:
    x = np.random.uniform(0, 1, (N, 2))
    count = np.sum(np.sum(x**2, axis=1) <= 1)
    pi_est = 4 * count / N
    return pi_est


if __name__ == "__main__":
    N_lst = [10, 100, 1000, 10000, 100000, 1000000]
    pi_est_mean_lst = []
    pi_est_std_lst = []
    error_mean_lst = []
    error_std_lst = []
    
    repeats = 10
    EPS = 1e-12
    
    for N in N_lst:
        
        pi_repeat_lst = []
        for i in range(repeats):
            pi_est = estimate_pi(N)
            pi_repeat_lst.append(pi_est)
        
        pi_est_mean_lst.append(np.mean(pi_repeat_lst))
        pi_est_std_lst.append(np.std(pi_repeat_lst))
        
        abs_errors = [np.abs(p - np.pi) + EPS for p in pi_repeat_lst]
        error_mean_lst.append(np.mean(abs_errors))
        error_std_lst.append(np.std(abs_errors))
    
    # TODO：将均值和标准差换为中位数和四分位数
    
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(N_lst, pi_est_mean_lst, yerr=2*np.array(pi_est_std_lst), fmt="o-", label="estimated $\pi$")
    plt.axhline(np.pi, color="k", linestyle="--", label=r"true $\pi$")
    plt.legend(loc="best")
    plt.xscale("log")
    plt.xlabel(r"sample size $N$")
    plt.ylabel(r"estimated $\pi$")
    
    plt.subplot(1, 2, 2)
    plt.errorbar(N_lst, error_mean_lst, yerr=2*np.array(error_std_lst), fmt="o-")
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"sample size $N$")
    plt.ylabel(r"estimation error")
    
    plt.tight_layout()
    plt.savefig("img/fig_monte_carlo_approximation.png", dpi=450)
    
    
