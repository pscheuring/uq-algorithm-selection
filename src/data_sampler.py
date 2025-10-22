from collections.abc import Callable
from typing import Callable, Dict, Union

import numpy as np
from scipy.special import expit, softplus
from scipy.stats import qmc


# 1NN-DTW
def f_cosine__1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.941927 - 0.110350 * (x1 + 3.316139) + 0.173276 * np.cos(x1 - 12.190082)
    )


def sigma_sinusoidal__1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.379897 - 0.054016 * (x1 - 24.542853) - 0.205750 * np.sin(x1 + 5.189004)
    )


# Arsenal
def f_cubic_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.848639 + 0.003304 * (x1 + 3.674640) - 0.005160 * (x1 + 2.824241) ** 3
    )


def sigma_cubic_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        1.585504 + 0.049867 * (x1 - 109.444859) - 0.001591 * (x1 + 0.328558) ** 3
    )


# BOSS
def f_quadratic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.506563 + 0.188097 * (x1 - 0.083922) + 0.049220 * (x1 - 1.215081) ** 2
    )


def sigma_quadratic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -66.201154 + 0.017781 * (x1 + 14.134653) - 0.003416 * (x1 - 1.716402) ** 2
    )


# CIF
def f_quadratic_cif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.230848 - 0.098095 * (x1 + 3.689441) + 0.013408 * (x1 - 0.384407) ** 2
    )


def sigma_quadratic_cif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.287526 + 0.030809 * (x1 - 0.460851) + 0.013363 * (x1 - 1.857690) ** 2
    )


# CNN
def f_quadratic_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.837361 + 0.153910 * (x1 + 2.533334) - 0.023559 * (x1 + 1.401457) ** 2
    )


def sigma_cubic_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        95.619976 - 39.445799 * (x1 + 3.751026) + 0.077166 * (x1 - 7.450403) ** 3
    )


# Catch22
def f_quadratic_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.641213 + 0.100582 * (x1 + 3.453945) - 0.007841 * (x1 + 2.519100) ** 2
    )


def sigma_cubic_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -170.064108 + 0.668128 * (x1 + 250.318879) - 0.020206 * (x1 + 0.462914) ** 3
    )


# DrCIF
def f_quadratic_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.632163 - 0.157558 * (x1 + 10.967733) + 0.017347 * (x1 + 1.938623) ** 2
    )


def sigma_cosine_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.967748 - 0.033118 * (x1 + 0.667889) + 0.024895 * np.cos(x1 + 13.517350)
    )


# EE
def f_cosine_ee_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.555834 - 0.078028 * (x1 - 0.051166) - 0.791565 * np.cos(x1 - 7.604298)
    )


def sigma_quadratic_ee_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -8.773922 - 0.245718 * (x1 - 11.011112) + 0.067859 * (x1 + 1.824243) ** 2
    )


# FreshPRINCE
def f_quadratic_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.598167 + 0.056364 * (x1 + 2.066425) + 0.028482 * (x1 - 1.729583) ** 2
    )


def sigma_quadratic_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.050999 - 0.006800 * (x1 + 7.394498) + 0.008203 * (x1 - 0.479792) ** 2
    )


# HC1
def f_cosine_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.570038 - 0.036011 * (x1 + 3.481986) + 0.983148 * np.cos(x1 - 174.125483)
    )


def sigma_cubic_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        4.082507 + 0.188011 * (x1 - 44.853839) - 0.006555 * (x1 - 0.439561) ** 3
    )


# HC2
def f_sinusoidal_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.724537 - 0.032206 * (x1 - 104.018276) + 1.046244 * np.sin(x1 + 3.353197)
    )


def sigma_cubic_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        36.815582 + 0.208746 * (x1 - 196.536302) - 0.006990 * (x1 - 0.171406) ** 3
    )


# Hydra-MR
def f_cosine_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        3.455785 - 0.109071 * (x1 + 7.509386) - 1.725167 * np.cos(x1 - 7.584948)
    )


def sigma_quadratic_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -6.992915 - 0.133198 * (x1 - 11.187717) + 0.049947 * (x1 + 1.598588) ** 2
    )


# Hydra
def f_cosine_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.665519 - 0.051857 * (x1 + 0.003295) - 1.500696 * np.cos(x1 - 7.582998)
    )


def sigma_cubic_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        40.537089 + 0.232768 * (x1 - 192.473754) - 0.007811 * (x1 - 0.280013) ** 3
    )


# InceptionT
def f_cubic_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        22.956243 + 0.094365 * (x1 - 223.547806) - 0.004167 * (x1 + 0.109153) ** 3
    )


def sigma_quadratic_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        1041.946753
        + 294.752648 * (x1 - 1041.176656)
        + 0.009997 * (x1 - 146.381271) ** 2
    )


# Mini-R
def f_cubic_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -1.470548 + 0.200284 * (x1 + 18.613424) - 0.010350 * (x1 + 1.334398) ** 3
    )


def sigma_cubic_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.720808 + 0.002931 * (x1 - 68.575978) - 0.000470 * (x1 + 3.905622) ** 3
    )


# MrSQM
def f_quadratic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.938416 + 0.117071 * (x1 - 0.318279) + 0.045723 * (x1 - 0.470811) ** 2
    )


def sigma_cubic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -15.432311 - 0.097523 * (x1 - 115.329633) + 0.001355 * (x1 + 1.900370) ** 3
    )


# Multi-R
def f_cosine_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.750147 - 0.024368 * (x1 + 3.560090) + 1.118282 * np.cos(x1 - 4.405538)
    )


def sigma_cubic_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        33.613107 + 0.292581 * (x1 - 129.120471) - 0.009577 * (x1 - 0.124035) ** 3
    )


# PF
def f_quadratic_pf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.887149 - 0.036904 * (x1 + 1.112891) + 0.012220 * (x1 - 2.128276) ** 2
    )


def sigma_cubic_pf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        8.915015 + 0.106974 * (x1 - 123.196687) - 0.001310 * (x1 - 3.837708) ** 3
    )


# RDST
def f_quadratic_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.868421 + 0.058588 * (x1 + 2.376990) + 0.009317 * (x1 + 0.268329) ** 2
    )


def sigma_cubic_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -82.529011 + 0.403300 * (x1 + 196.002470) - 0.012444 * (x1 + 0.456500) ** 3
    )


# RISE
def f_quadratic_rise_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.004484 - 0.075751 * (x1 + 5.730618) + 0.020321 * (x1 + 0.336067) ** 2
    )


def sigma_quadratic_rise_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -7.654294 - 0.110745 * (x1 - 30.208511) + 0.010019 * (x1 + 3.785238) ** 2
    )


# RIST
def f_sinusoidal_rist_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.885495 + 0.228740 * (x1 + 1.371575) - 0.888073 * np.sin(x1 - 0.329287)
    )


def sigma_cosine_rist_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -12.717406 - 0.305172 * (x1 - 22.181533) + 2.218927 * np.cos(x1 + 17.109064)
    )


# ROCKET
def f_cubic_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.782751 + 0.016232 * (x1 + 8.792844) - 0.005118 * (x1 + 2.697997) ** 3
    )


def sigma_cubic_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.445779 + 0.055372 * (x1 + 28.387466) - 0.002027 * (x1 + 0.537528) ** 3
    )


# RSF
def f_cosine_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.108027 + 0.048462 * (x1 + 36.926736) - 0.101842 * np.cos(x1 - 11.885046)
    )


def sigma_cubic_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        6.895344 + 0.667401 * (x1 - 14.795979) - 0.020849 * (x1 + 0.399996) ** 3
    )


# RSTSF
def f_quadratic_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.201644 - 0.115047 * (x1 + 2.544427) + 0.022101 * (x1 + 1.046055) ** 2
    )


def sigma_cubic_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -9.080911 - 0.102445 * (x1 - 45.729959) + 0.000585 * (x1 + 7.556147) ** 3
    )


# ResNet
def f_sinusoidal_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.048669 + 0.047341 * (x1 + 3.860433) + 0.450568 * np.sin(x1 + 3.841300)
    )


def sigma_quadratic_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -39.631663 - 1.343598 * (x1 + 26.491297) + 2.005402 * (x1 + 0.295929) ** 2
    )


# STC
def f_sinusoidal_stc_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.465523 - 0.018600 * (x1 + 10.827333) - 0.925655 * np.sin(x1 + 0.229552)
    )


def sigma_cubic_stc_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        1.019365 + 0.175157 * (x1 - 28.672872) - 0.006864 * (x1 - 0.176325) ** 3
    )


# STSF
def f_sinusoidal_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.566831 - 0.093057 * (x1 - 5.721057) + 0.325630 * np.sin(x1 + 7.515832)
    )


def sigma_cubic_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        13.095024 - 0.097476 * (x1 + 178.129976) + 0.000599 * (x1 + 6.790582) ** 3
    )


# ShapeDTW
def f_cosine_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        4.561438 - 0.114212 * (x1 + 35.962048) - 0.152517 * np.cos(x1 - 9.616464)
    )


def sigma_quadratic_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.898342 - 0.100019 * (x1 - 14.190469) + 0.017924 * (x1 + 1.404359) ** 2
    )


# Signatures
def f_quadratic_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.338797 - 0.146843 * (x1 - 0.590688) + 0.008810 * (x1 + 0.241508) ** 2
    )


def sigma_cubic_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        44.419883 + 0.703427 * (x1 - 87.777153) - 0.000088 * (x1 - 52.870884) ** 3
    )


# TDE
def f_quadratic_tde_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.948131 - 0.121783 * (x1 + 2.234514) + 0.044937 * (x1 + 2.052517) ** 2
    )


def sigma_cosine_tde_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.123897 - 0.064370 * (x1 - 0.163455) - 0.083439 * np.cos(x1 - 5.347431)
    )


# TS-CHIEF
def f_cosine_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.230789 - 0.069500 * (x1 - 34.759744) - 1.557389 * np.cos(x1 - 7.628756)
    )


def sigma_cubic_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -1.085183 + 0.187420 * (x1 - 17.882339) - 0.006344 * (x1 - 0.539172) ** 3
    )


# TSF
def f_quadratic_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.812016 - 0.071302 * (x1 + 4.249364) + 0.012470 * (x1 - 1.715980) ** 2
    )


def sigma_quadratic_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.218871 - 0.005855 * (x1 + 10.410147) + 0.015457 * (x1 - 0.906873) ** 2
    )


# TSFresh
def f_cosine_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.381371 + 0.155577 * (x1 - 0.489518) + 0.364799 * np.cos(x1 - 2.458412)
    )


def sigma_cubic_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        81.194834 + 1.920251 * (x1 - 39.043689) - 6.762968 * (x1 + 5.221846) ** 3
    )


# WEASEL-D
def f_quadratic_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.319604 - 0.210755 * (x1 - 4.278125) + 0.043832 * (x1 + 3.126242) ** 2
    )


def sigma_sinusoidal_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.317144 - 0.036756 * (x1 + 21.777535) - 0.005568 * np.sin(x1 - 3.084819)
    )


# WEASEL
def f_quadratic_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.960802 - 0.053526 * (x1 + 1.129108) + 0.012543 * (x1 - 2.305353) ** 2
    )


def sigma_cubic_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        5.683443 + 0.113077 * (x1 - 84.485574) - 0.000879 * (x1 - 3.811410) ** 3
    )


# cBOSS
def f_quadratic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.413710 + 0.316925 * (x1 + 2.350191) + 0.044110 * (x1 - 2.836558) ** 2
    )


def sigma_quadratic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -2.047815 - 0.081492 * (x1 + 27.198818) + 0.008365 * (x1 + 1.933185) ** 2
    )


### 2 FEATURES ###
def f_quadratic_rocket_2_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]
    return expit(
        1.330836 - 0.04011 * (x1 + 10.326781) + 0.234344 * (x2 - 0.244166) ** 2
    )


def sigma_cosine_rocket_2_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]
    return softplus(
        -6.172266 - 0.066204 * (x1 - 31.134696) + 0.399073 * np.cos(x2 + 3.088828)
    )


# HC2 - 2 Feat
def f_quadratic_hc2_2_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    return expit(
        4.956188 + 0.160603 * (x1 - 17.817124) + 0.038322 * (x2 + 0.579638) ** 2
    )


def sigma_quadratic_hc2_2_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    return softplus(
        -5.330938 - 0.105839 * (x1 - 1.965741) + 0.002646 * (x2 - 9.524848) ** 2
    )


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "f_cosine__1nn_dtw_1_feat": f_cosine__1nn_dtw_1_feat,
    "f_cubic_arsenal_1_feat": f_cubic_arsenal_1_feat,
    "f_quadratic_boss_1_feat": f_quadratic_boss_1_feat,
    "f_quadratic_cif_1_feat": f_quadratic_cif_1_feat,
    "f_quadratic_cnn_1_feat": f_quadratic_cnn_1_feat,
    "f_quadratic_catch22_1_feat": f_quadratic_catch22_1_feat,
    "f_quadratic_drcif_1_feat": f_quadratic_drcif_1_feat,
    "f_cosine_ee_1_feat": f_cosine_ee_1_feat,
    "f_quadratic_freshprince_1_feat": f_quadratic_freshprince_1_feat,
    "f_cosine_hc1_1_feat": f_cosine_hc1_1_feat,
    "f_sinusoidal_hc2_1_feat": f_sinusoidal_hc2_1_feat,
    "f_cosine_hydra_mr_1_feat": f_cosine_hydra_mr_1_feat,
    "f_cosine_hydra_1_feat": f_cosine_hydra_1_feat,
    "f_cubic_inceptiont_1_feat": f_cubic_inceptiont_1_feat,
    "f_cubic_mini_r_1_feat": f_cubic_mini_r_1_feat,
    "f_quadratic_mrsqm_1_feat": f_quadratic_mrsqm_1_feat,
    "f_cosine_multi_r_1_feat": f_cosine_multi_r_1_feat,
    "f_quadratic_pf_1_feat": f_quadratic_pf_1_feat,
    "f_quadratic_rdst_1_feat": f_quadratic_rdst_1_feat,
    "f_quadratic_rise_1_feat": f_quadratic_rise_1_feat,
    "f_sinusoidal_rist_1_feat": f_sinusoidal_rist_1_feat,
    "f_cubic_rocket_1_feat": f_cubic_rocket_1_feat,
    "f_cosine_rsf_1_feat": f_cosine_rsf_1_feat,
    "f_quadratic_rstsf_1_feat": f_quadratic_rstsf_1_feat,
    "f_sinusoidal_resnet_1_feat": f_sinusoidal_resnet_1_feat,
    "f_sinusoidal_stc_1_feat": f_sinusoidal_stc_1_feat,
    "f_sinusoidal_stsf_1_feat": f_sinusoidal_stsf_1_feat,
    "f_cosine_shapedtw_1_feat": f_cosine_shapedtw_1_feat,
    "f_quadratic_signatures_1_feat": f_quadratic_signatures_1_feat,
    "f_quadratic_tde_1_feat": f_quadratic_tde_1_feat,
    "f_cosine_ts_chief_1_feat": f_cosine_ts_chief_1_feat,
    "f_quadratic_tsf_1_feat": f_quadratic_tsf_1_feat,
    "f_cosine_tsfresh_1_feat": f_cosine_tsfresh_1_feat,
    "f_quadratic_weasel_d_1_feat": f_quadratic_weasel_d_1_feat,
    "f_quadratic_weasel_1_feat": f_quadratic_weasel_1_feat,
    "f_quadratic_cboss_1_feat": f_quadratic_cboss_1_feat,
    "f_quadratic_hc2_2_feat": f_quadratic_hc2_2_feat,
}

SIGMAS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigma_sinusoidal__1nn_dtw_1_feat": sigma_sinusoidal__1nn_dtw_1_feat,
    "sigma_cubic_arsenal_1_feat": sigma_cubic_arsenal_1_feat,
    "sigma_quadratic_boss_1_feat": sigma_quadratic_boss_1_feat,
    "sigma_quadratic_cif_1_feat": sigma_quadratic_cif_1_feat,
    "sigma_cubic_cnn_1_feat": sigma_cubic_cnn_1_feat,
    "sigma_cubic_catch22_1_feat": sigma_cubic_catch22_1_feat,
    "sigma_cosine_drcif_1_feat": sigma_cosine_drcif_1_feat,
    "sigma_quadratic_ee_1_feat": sigma_quadratic_ee_1_feat,
    "sigma_quadratic_freshprince_1_feat": sigma_quadratic_freshprince_1_feat,
    "sigma_cubic_hc1_1_feat": sigma_cubic_hc1_1_feat,
    "sigma_cubic_hc2_1_feat": sigma_cubic_hc2_1_feat,
    "sigma_quadratic_hydra_mr_1_feat": sigma_quadratic_hydra_mr_1_feat,
    "sigma_cubic_hydra_1_feat": sigma_cubic_hydra_1_feat,
    "sigma_quadratic_inceptiont_1_feat": sigma_quadratic_inceptiont_1_feat,
    "sigma_cubic_mini_r_1_feat": sigma_cubic_mini_r_1_feat,
    "sigma_cubic_mrsqm_1_feat": sigma_cubic_mrsqm_1_feat,
    "sigma_cubic_multi_r_1_feat": sigma_cubic_multi_r_1_feat,
    "sigma_cubic_pf_1_feat": sigma_cubic_pf_1_feat,
    "sigma_cubic_rdst_1_feat": sigma_cubic_rdst_1_feat,
    "sigma_quadratic_rise_1_feat": sigma_quadratic_rise_1_feat,
    "sigma_cosine_rist_1_feat": sigma_cosine_rist_1_feat,
    "sigma_cubic_rocket_1_feat": sigma_cubic_rocket_1_feat,
    "sigma_cubic_rsf_1_feat": sigma_cubic_rsf_1_feat,
    "sigma_cubic_rstsf_1_feat": sigma_cubic_rstsf_1_feat,
    "sigma_quadratic_resnet_1_feat": sigma_quadratic_resnet_1_feat,
    "sigma_cubic_stc_1_feat": sigma_cubic_stc_1_feat,
    "sigma_cubic_stsf_1_feat": sigma_cubic_stsf_1_feat,
    "sigma_quadratic_shapedtw_1_feat": sigma_quadratic_shapedtw_1_feat,
    "sigma_cubic_signatures_1_feat": sigma_cubic_signatures_1_feat,
    "sigma_cosine_tde_1_feat": sigma_cosine_tde_1_feat,
    "sigma_cubic_ts_chief_1_feat": sigma_cubic_ts_chief_1_feat,
    "sigma_quadratic_tsf_1_feat": sigma_quadratic_tsf_1_feat,
    "sigma_cubic_tsfresh_1_feat": sigma_cubic_tsfresh_1_feat,
    "sigma_sinusoidal_weasel_d_1_feat": sigma_sinusoidal_weasel_d_1_feat,
    "sigma_cubic_weasel_1_feat": sigma_cubic_weasel_1_feat,
    "sigma_quadratic_cboss_1_feat": sigma_quadratic_cboss_1_feat,
    "sigma_quadratic_hc2_2_feat": sigma_quadratic_hc2_2_feat,
}


class DataSampler:
    """Config-driven sampler for synthetic regression datasets (single- or multi-output).

    This class generates training data via Latin Hypercube Sampling (optionally from
    two disjoint intervals) with heteroscedastic noise, and test data on a uniform
    Cartesian grid.

    Expected job schema (flat dict):
        seed (int): RNG seed for reproducibility.
        function (list[str, str]): ["function_name", "sigma_name"].
        train_interval (list[float] | list[list[float, float]]):
            - [a, b] for a single interval per feature, or
            - [[a, b], [c, d]] to draw half the samples from [a, b] and half from [c, d].
        train_instances (int): Number of unique training inputs (before repeats).
        train_repeats (int): How many times to repeat each unique input (duplicates rows).
        test_interval (list[float, float]): [min, max] range per feature for the test grid.
        test_grid_length (int): Number of grid points per feature for the test set.
        add_points (optional, list[float]): Extra 1D points; broadcast to all dims if D>1.

    Constructor arguments:
        job (dict): The configuration dictionary described above.
        functions (dict[str, Callable[[np.ndarray], np.ndarray]]):
            Maps function names to target-generating callables f(X)->y_clean.
        sigmas (dict[str, Callable[[np.ndarray], np.ndarray]]):
            Maps noise names to callables sigma(X)->σ (per-sample or per-output std).

    Notes:
        - Input dimensionality D is inferred by probing functions[function_name].
        - Noise is added as y = y_clean + Normal(0, sigma).
        - Test y is noise-free (clean). Duplicates in the test grid (including extra points)
          are removed.
        - When using two train intervals, samples per feature are drawn from both intervals
          and independently permuted to avoid grid artifacts.
    """

    def __init__(
        self,
        job: dict[str, str | int | float | list],
        functions: dict[str, Callable] = FUNCTIONS,
        sigmas: dict[str, Callable] = SIGMAS,
    ) -> None:
        """Initialize the sampler.

        Args:
            job: Flat config with data generation settings.
            functions: Mapping function_name -> f(X) -> y_clean.
            sigmas: Mapping noise_name -> sigma(X) -> σ.
        """
        self.job = job
        self.functions = functions
        self.sigmas = sigmas
        self.rng = np.random.default_rng(seed=int(job["seed"]))

    def sample_train_data(self) -> dict[str, np.ndarray]:
        """Sample training data with heteroscedastic noise using Latin Hypercube Sampling.

        Returns:
            dict with:
                - X: (N, D) inputs
                - y: (N,) or (N,1) noisy targets
                - y_clean: (N,) or (N,1) noise-free targets
                - sigma: (N,) or (N,1) noise std per sample
                - n_features: scalar number of input features D
        """
        fn = self.functions[self.job["function"][0]]
        sigma_fn = self.sigmas[self.job["function"][1]]

        interval_spec = self.job["train_interval"]
        n_instances = int(self.job["train_instances"])
        n_repeats = int(self.job["train_repeats"])

        n_features = self._probe_dim(fn)
        is_single_interval = not isinstance(interval_spec[0], (list, tuple, np.ndarray))

        if is_single_interval:
            a, b = map(float, interval_spec)
            sampler = qmc.LatinHypercube(d=n_features, seed=self.rng)
            X_unique = qmc.scale(
                sampler.random(n_instances),
                [a] * n_features,
                [b] * n_features,
            )
        else:
            (a, b), (c, d) = interval_spec
            a, b, c, d = map(float, (a, b, c, d))
            n1 = n_instances // 2
            n2 = n_instances - n1

            if n_features == 1:
                s1 = qmc.LatinHypercube(d=1, seed=self.rng)
                s2 = qmc.LatinHypercube(d=1, seed=self.rng)
                axis = np.concatenate(
                    [
                        qmc.scale(s1.random(n1), [a], [b]).ravel(),
                        qmc.scale(s2.random(n2), [c], [d]).ravel(),
                    ]
                )
                X_unique = axis.reshape(-1, 1)
            else:
                # Per feature: draw LHS samples from both intervals and independently permute.
                axes: list[np.ndarray] = []
                for _ in range(n_features):
                    s1 = qmc.LatinHypercube(d=1, seed=self.rng)
                    s2 = qmc.LatinHypercube(d=1, seed=self.rng)
                    part1 = qmc.scale(s1.random(n1), [a], [b]).ravel()
                    part2 = qmc.scale(s2.random(n2), [c], [d]).ravel()
                    base_axis = np.concatenate([part1, part2])
                    axes.append(base_axis[self.rng.permutation(n_instances)])
                X_unique = np.stack(axes, axis=1)

        # Repeats (repeat rows of X_unique)
        X = (
            X_unique
            if n_repeats < 1
            # add +1, because repeats=1 is equivalent to no repeats
            else np.repeat(X_unique, repeats=n_repeats + 1, axis=0)
        )

        # Targets + heteroscedastic noise
        y_clean = fn(X)  # (N,) or (N,1)
        sigma = sigma_fn(X)  # (N,) or (N,1)
        noise = self.rng.normal(0.0, sigma)
        y = y_clean + noise

        return {
            "X": X,
            "y": y,
            "sigma": sigma,
            "n_features": np.int64(n_features),
        }

    def sample_test_data(self) -> dict[str, np.ndarray]:
        """Sample test data on a uniform Cartesian grid spanning `test_interval`.

        Builds a grid with test_grid_length points per dimension.
        Optionally appends 1D add_points (broadcast across dims when D>1).
        Duplicate rows are removed.

        Returns:
            dict with:
                - X: (M, D) test inputs
                - y: (M,) or (M,1) clean targets
                - sigma: (M,) or (M,1) noise std per input
        """
        fn = self.functions[self.job["function"][0]]
        sigma_fn = self.sigmas[self.job["function"][1]]
        n_features = self._probe_dim(fn)

        lo, hi = map(float, self.job["test_interval"])
        grid_length = int(self.job["test_grid_length"])
        add_points = self.job["test_add_points"]

        axis = np.linspace(lo, hi, grid_length, dtype=np.float64)
        mesh = np.meshgrid(*([axis] * n_features), indexing="xy")
        X_grid = np.stack(mesh, axis=-1).reshape(-1, n_features)

        X_list: list[np.ndarray] = [X_grid]

        if add_points is not None:
            # Expecects nested list: [[...]]
            vals = np.asarray(add_points, dtype=np.float64).reshape(-1, 1)
            X_extra = vals if n_features == 1 else np.tile(vals, (1, n_features))
            X_list.append(X_extra)

        X = np.unique(np.vstack(X_list), axis=0)
        y = fn(X)  # clean
        sigma = sigma_fn(X)  # noise std

        return {"X": X, "y": y, "sigma": sigma}

    def _probe_dim(self, f: Callable[[np.ndarray], np.ndarray]) -> int:
        """Infer input dimensionality D of `f` by probing shapes (N, D) for D=1..1000.

        Args:
            f: Callable mapping (N, D) arrays to targets.

        Returns:
            Detected input dimension D.

        Raises:
            ValueError: If no dimensionality in [1, 1000] is accepted.
        """
        for d in range(1, 1001):
            try:
                _ = f(np.zeros((1, d)))
                return d
            except Exception:
                continue
        raise ValueError("Could not infer input dimensionality D in range [1, 1000].")
