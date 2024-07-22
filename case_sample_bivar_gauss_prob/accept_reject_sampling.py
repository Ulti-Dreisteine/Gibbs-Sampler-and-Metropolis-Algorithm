# -*- coding: utf-8 -*-
"""
Created on 2024/07/21 12:52:54

@File -> accept_reject_sampling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 接受拒绝采样
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from case_sample_bivar_gauss_prob.bivar_mixed_gaussian_pdf import cal_merged_gauss_pdf

if __name__ == "__main__":
    # 待采样分布参数
    mu_lst = [np.array([-3, -3]), np.array([3, 3])]
    sigma_lst = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
    weights = [0.2, 0.8]
    
    # 采样边界参数
    x_bounds = [-10, 10]
    y_bounds = [-10, 10]
    
    n_loops = 10000
    plt.figure(figsize = (10, 10))
    for idx, M in enumerate([1e-8, 1, 1e1, 1e2]):
        x_samples_lst = []
        y_samples_lst = []
        
        for _ in range(n_loops):
            # 设提议分布 Q(x, y) 为均匀分布，根据提议分布采样
            x = np.random.uniform(x_bounds[0], x_bounds[1])
            y = np.random.uniform(y_bounds[0], y_bounds[1])
            
            # 计算目标分布 P(x, y)
            p = cal_merged_gauss_pdf(x, y, mu_lst, sigma_lst, weights)
            
            # 计算提议分布 Q(x, y)
            q = 1 / ((x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]))
            
            # 计算接受概率
            alpha = p / (M * q)
            
            # 接受拒绝
            if np.random.rand() < alpha:
                x_samples_lst.append(x)
                y_samples_lst.append(y)
                
        # 绘制采样结果
        
        # 待采样分布
        x = np.linspace(x_bounds[0], x_bounds[1], 100)
        y = np.linspace(y_bounds[0], y_bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = cal_merged_gauss_pdf(X[i, j], Y[i, j], mu_lst, sigma_lst, weights)
        
        plt.subplot(2, 2, idx + 1)
        plt.title(f"$M$ = {M}")
        plt.contourf(X, Y, Z, 50, cmap="Reds")
        plt.xlabel("$x$")
        
        if idx == 0:
            plt.ylabel("$y$")
        
        # if idx == 2:
        #     plt.colorbar(label="Prob. Dens.")
        
        plt.scatter(x_samples_lst, y_samples_lst, marker="o", c = "k", s = 1, linewidths=0.2, alpha=1, zorder=1)
        plt.xlim(x_bounds)
        plt.ylim(y_bounds)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        # plt.show()
        
        # 在图上注明总循环次数和有效采样点数目
        plt.text(0.6, 0.95, f"total loops: {n_loops}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.6, 0.9, f"valid samples: {len(x_samples_lst)}", fontsize=12, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig("asset/案例_接受拒绝采样.png", dpi=300)
    
    