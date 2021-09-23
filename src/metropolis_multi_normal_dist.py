# -*- coding: utf-8 -*-
"""
Created on 2021/09/22 14:37:43

@File -> metropolis_hastings_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Metropolis Hastings采样算法测试
"""

__doc__ = """
    分别采用Metropolis和MH算法，使用均匀分布对多个高斯分布的融合分布进行采样
    Metropolis算法和MH算法都只需要知道样本对应概率的相对关系便可进行采样, 不需要目标分布的全局信息
"""

import numpy as np 
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 2))
sys.path.append(BASE_DIR)

from src.settings import plt

EPS = 1e-12


# 高斯分布.

def calGaussianPDF(x, mu, sigma):
    """一元高斯分布概率密度函数"""
    a = 1 / (np.sqrt(2 * np.pi) * sigma)
    b = -np.power(x - mu, 2) / (2 * sigma**2)
    return a * np.exp(b)


def calMergedGaussianPDF(x_s, gaussian_params_a, gaussian_params_b, w_a, w_b):
    """在同一x处, 两种分布加权和。注意该值未归一化, 因此不是真正的PDF, 但可以用于Metropolis采样"""
    return w_a * calGaussianPDF(x_s, *gaussian_params_a) + w_b * calGaussianPDF(x_s, *gaussian_params_b)


# 样本指标计算.

def sample() -> float:
    """从一个均匀分布中进行采样"""
    return (np.random.random() - 0.5) * 20


def calTicksAndPDF(S, bins: int):
    freqs, edges = np.histogram(S, bins = bins)
    ticks = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    probs = freqs / np.sum(freqs)
    pdfs = probs / (ticks[1] - ticks[0])
    return ticks, pdfs



if __name__ == '__main__':
    
    # ---- 目标分布参数 -----------------------------------------------------------------------------

    gaussian_params_a = [-3.0, 1.0]
    gaussian_params_b = [3.0, 1.0]

    x_pdf = np.linspace(-10, 10, 100)
    y_pdf_a = calGaussianPDF(x_pdf, *gaussian_params_a)
    y_pdf_b = calGaussianPDF(x_pdf, *gaussian_params_b)

    # 两种概率分布的权重.
    w_a, w_b = 1.0, 5.0

    # 求得归一化后的融合概率分布.
    y_pdf_union =  w_a * y_pdf_a + w_b * y_pdf_b
    y_pdf_union /= np.sum(y_pdf_union * (x_pdf[1] - x_pdf[0]))

    # ---- 进行采样 --------------------------------------------------------------------------------

    x_s = sample()
    pi_s = calMergedGaussianPDF(x_s, gaussian_params_a, gaussian_params_b, w_a, w_b)
    S = [x_s]
    Pi = [pi_s]
    
    i = 0
    N = 10000
    while True:
        # 采样, 作为候选.
        x_c = sample()

        # 计算新样本概率密度.
        pi_c = calMergedGaussianPDF(x_c, gaussian_params_a, gaussian_params_b, w_a, w_b)

        # 接受或拒绝.
        alpha = np.min([1, pi_c / (Pi[-1] + EPS)])  # 对于本例MH算法与Metropolis算法是一样的

        r = np.random.random()

        if r <= alpha: 
            x_s, pi_s = x_c, pi_c
        else: 
            x_s, pi_s = S[-1], Pi[-1]

        # 样本链增长.
        S.append(x_s)
        Pi.append(pi_s)

        i += 1
        if i == N:
            break
    
    # ---- 画图 ------------------------------------------------------------------------------------
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(S)
    axs[0].set_title('Metropolis Sampling Process', fontsize = 14)
    axs[0].set_xlabel('iteration number')
    axs[0].set_ylabel('value')

    ticks, pdfs = calTicksAndPDF(S[:], bins = 50)

    axs[1].plot(ticks, pdfs)
    axs[1].plot(x_pdf, y_pdf_union, '--')
    axs[1].set_xlim([-10.0, 10.0])
    axs[1].set_title('Distribution: Sampling vs Theoretical', fontsize = 14)
    axs[1].set_xlabel('value')
    axs[1].set_ylabel('PDF')

    fig.tight_layout()
    


    