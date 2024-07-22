# -*- coding: utf-8 -*-
"""
Created on 2024/07/21 10:46:40

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 对二元高斯分布进行采样
"""

from typing import List
import numpy as np
import warnings
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


# 二元高斯分布函数
def cal_bivar_gauss_pdf(x: float, y: float, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    计算二元高斯分布概率密度函数
    
    Params:
    -------
    x: x坐标
    y: y坐标
    mu: 均值向量，shape = (2,)
    sigma: 协方差矩阵，对称正定, shape = (2, 2)
    """
    x_mu = x - mu[0]
    y_mu = y - mu[1]
    cov = sigma[0, 1]
    sigma_x = sigma[0, 0]
    sigma_y = sigma[1, 1]
    z = x_mu ** 2 / sigma_x + y_mu ** 2 / sigma_y - 2 * cov * x_mu * y_mu
    return np.exp(-z / (2 * (1 - cov ** 2))) / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - cov ** 2))


# 计算多个高斯分布的加权分布
def cal_merged_gauss_pdf(x: float, y: float, mu_lst: List[np.ndarray], sigma_lst: List[np.ndarray], 
                         weights: List[float]) -> float:
    """
    计算多个高斯分布的加权融合分布
    
    Params:
    -------
    x: x坐标
    mu_lst: 均值向量列表, 每个 mu 的 shape = (2, )
    sigma_lst: 协方差矩阵列表, 每个 sigma 的 shape = (2, 2)
    weights: 权重列表
    """
    assert len(mu_lst) == len(sigma_lst) == len(weights)
    
    try:
        assert sum(weights) == 1
    except Exception as _:
        warnings.warn("The sum of weights is not equal to 1")
        weights = [w / sum(weights) for w in weights]
        
    return sum(
        weights[i] * cal_bivar_gauss_pdf(x, y, mu_lst[i], sigma_lst[i])
        for i in range(len(mu_lst))
    )

if __name__ == "__main__":
    
    # ---- 单个二元高斯分布 --------------------------------------------------------------------------
    
    # 绘制二元高斯分布概率密度函数
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    mu = np.array([1, 1])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    Z = cal_bivar_gauss_pdf(X, Y, mu, sigma)
            
    # plt.figure(figsize=(5, 5))
    # plt.contourf(X, Y, Z, 50)
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    
    # ---- 多个二元高斯分布的加权融合 ------------------------------------------------------------------
    
    mu_lst = [np.array([-3, -3]), np.array([3, 3])]
    sigma_lst = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
    weights = [0.2, 0.8]
    
    # 绘制多个二元高斯分布的加权融合概率密度函数
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = cal_merged_gauss_pdf(X[i, j], Y[i, j], mu_lst, sigma_lst, weights)
        
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, 50, cmap="Reds")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(label="Prob. Dens.")
    plt.savefig("asset/bivar_mixed_gaussian_pdf.png", dpi=450)
    