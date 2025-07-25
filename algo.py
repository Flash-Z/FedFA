import os
from matplotlib import pyplot as plt
import numpy as np
import math

def cal_g(alpha, beta, Lm, local_epoch):
    W = 1 + 3*beta + alpha*Lm
    X = np.sqrt((1 + beta + alpha*Lm)**2 + 4*beta)
    Y = 2*beta + alpha*Lm
    Z = 1 + beta + alpha*Lm
    
    # 检查nan
    assert not np.isnan(W)
    assert not np.isnan(X)
    assert not np.isnan(Y)
    assert not np.isnan(Z)
    
    term1 = (W + X)/(2 * X * Y) * ((X + Z)/2)**local_epoch
    term2 = (W - X)/(2 * X * Y) * ((Z - X)/2)**local_epoch
    term3 = 1/Y
    
    return term1 - term2 - term3

def cal_epsilon(
    alpha: float,
    gamma_k_minus_1: float,
    client_weights: float,
    local_epoch: float,
    lambda_k: float, # Note: Cannot use 'lambda' as it's a Python keyword
    zeta_k: float,
    Delta_c_minus_1: float,
    rho_k_m: float,
    sigma_k_m: float,
    g_of_mu_k_m: float # Represents the result of g^m_{[k]}(\mu^m_{[k]})
) -> float:
    """
    Calculates epsilon_0' based on the provided formula.

    Args:
        alpha: The alpha parameter (α).
        gamma_k_minus_1: The Gamma parameter from the previous step (Γ_{k-1}).
        d_m: The d_m parameter.
        local_epoch: The local_epoch parameter for step k.
        lambda_k_m: The lambda parameter for step k, component m (λ^m_{[k]}).
        zeta_k: The zeta parameter for step k (ζ_{[k]}).
        sigma_k_minus_1: The sigma parameter from the previous step (σ_{k-1}).
        rho_k_m: The rho parameter for step k, component m (ρ^m_{[k]}).
        sigma_k_m: The sigma parameter for step k, component m (σ^m_{[k]}).
        g_of_mu_k_m: The result of the function g evaluated at mu_k_m (g^m_{[k]}(\mu^m_{[k]})).

    Returns:
        The calculated value of epsilon_0'.

    Raises:
        ValueError: If the denominator is zero or the value inside the
                    square root is negative.
    """

    # Calculate the term that appears in both numerator (inside sqrt) and denominator
    term1 = gamma_k_minus_1 + client_weights * local_epoch * lambda_k * zeta_k

    # Calculate the denominator
    denominator = 2 * term1
    if denominator == 0:
        raise ValueError("Denominator cannot be zero.")

    # Calculate the second term multiplying 4*alpha inside the square root
    term2 =  Delta_c_minus_1 + rho_k_m * sigma_k_m * g_of_mu_k_m

    # Calculate the value inside the square root
    inside_sqrt = 1 + 4 * alpha * term1 * term2
    if inside_sqrt < 0:
        raise ValueError(f"Cannot take the square root of a negative number: {inside_sqrt}")

    # Calculate the numerator
    numerator = 1 + math.sqrt(inside_sqrt)

    # Calculate the final result
    epsilon_0_prime = numerator / denominator

    return epsilon_0_prime

def cal_target(epsilon, alpha, client_weight, sigma_m, rho_m, g_m):
    return epsilon + alpha * client_weight * sigma_m * rho_m * g_m

def cal_zeta(alpha, L):
    return alpha * (1 - alpha * L / 2)

def liner_search(alpha, beta, L, L_m, client_weight, sigma_m, rho_m, max_local_epoch, gamma_k_minus_1, Delta_c_minus_1, lambda_k):
    """在 [1, max_local_epoch_list] 区间进行线性搜索使得target最小的local_epoch值，只搜索整数
    """
    # 计算ζ
    zeta = cal_zeta(alpha, L)
    
    best_target = float('inf')
    best_local_epoch = None
    for local_epoch in range(1, max(max_local_epoch + 1, 2)):
        # 计算g_m
        g_m = cal_g(alpha, beta, L_m, local_epoch)
        # 计算epsilon
        epsilon = cal_epsilon(alpha, gamma_k_minus_1, client_weight, local_epoch, lambda_k, zeta, Delta_c_minus_1, rho_m, sigma_m, g_m)
        # 计算target
        target = cal_target(epsilon, alpha, client_weight, sigma_m, rho_m, g_m)
        if target < best_target:
            best_target = target
            best_local_epoch = local_epoch
    
    print(f"Best local epoch: {best_local_epoch}, Best target: {best_target}")
    
    return best_local_epoch

# ======================================================

def cal_Delta_c(Delta_c_minus_1, device_weights, rho_list, sigma_m_list, tau_d_list):
    """计算Delta_c"""
    return Delta_c_minus_1 + sum([dm * rho_d * sigma_m * tau_d for dm, rho_d, sigma_m, tau_d in zip(device_weights, rho_list, sigma_m_list, tau_d_list)])

def cal_Gamma_c(Gamma_c_minus_1, device_weights, lambda_k, local_epoch_list, zeta_k):
    """计算Gamma_c"""
    return Gamma_c_minus_1 + sum([dm * local_epoch * lambda_k * zeta_k for dm, local_epoch in zip(device_weights, local_epoch_list)])

def cal_LAMBDA_c(LAMBDA_c_minus_1, TAU_list):
    """计算LAMBDA_c, """
    return LAMBDA_c_minus_1 + max(TAU_list)

def cal_DGL(Delta_c_minus_1, Gamma_c_minus_1, LAMBDA_c_minus_1, device_weights, rho_list, sigma_m_list, alpha, beta, lambda_k, TAU_list, L_m_list, L, local_epoch_list):
    """∆ Γ Λ"""
    g_m_list = [cal_g(alpha, beta, Lm, local_epoch) for Lm, local_epoch in zip(L_m_list, local_epoch_list)]
    zeta = cal_zeta(alpha, L)
    
    Delta_c = cal_Delta_c(Delta_c_minus_1, device_weights, rho_list, sigma_m_list, g_m_list)
    Gamma_c = cal_Gamma_c(Gamma_c_minus_1, device_weights, lambda_k, local_epoch_list, zeta)
    LAMBDA_c = cal_LAMBDA_c(LAMBDA_c_minus_1, TAU_list)
    return Delta_c, Gamma_c, LAMBDA_c
    