from collections.abc import Callable

import numpy as np
from scipy.special import expit, softplus
from scipy.stats import qmc


### 1 FEATURE ###
# 1NN-DTW
def f_exponential__1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(-0.196845 - 0.157658 * (x1 - 3.717433) + 0.001721 * np.exp(x1 + 0.674))


def sigma_cubic__1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -11.573856 + 0.089195 * (x1 + 79.579092) - 0.00115 * (x1 - 4.801252) ** 3
    )


# Arsenal
def f_exponential_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(1.973566 + 0.00254 * (x1 + 4.521214) - 0.032041 * np.exp(x1 + 0.08652))


def sigma_cubic_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -54.086517 + 0.049867 * (x1 + 1006.965114) - 0.001591 * (x1 + 0.328552) ** 3
    )


# BOSS
def f_cubic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -19.982124 + 5.015953 * (x1 - 29.398752) - 0.000159 * (x1 - 101.926779) ** 3
    )


def sigma_quadratic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -2.975511 - 0.121306 * (x1 + 13.875194) + 0.002152 * (x1 + 17.457081) ** 2
    )


# CIF
def f_cosine_cif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -1.979111 - 0.103916 * (x1 - 29.807034) + 0.229794 * np.cos(x1 - 0.461136)
    )


def sigma_cubic_cif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.364636 + 0.956543 * (x1 - 24.975909) - 5.9e-05 * (x1 - 73.943529) ** 3
    )


# CNN
def f_cosine_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.518455 + 0.081001 * (x1 - 8.865435) - 0.460887 * np.cos(x1 - 0.434805)
    )


def sigma_cubic_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.757417 + 0.018735 * (x1 + 35.214265) - 0.001796 * (x1 - 2.380216) ** 3
    )


# Catch22
def f_cubic_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(0.887057 - 0.159107 * (x1 - 1.65488) + 0.006084 * (x1 + 0.82213) ** 3)


def sigma_cubic_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.71284 + 0.668128 * (x1 + 4.331308) - 0.020206 * (x1 + 0.462914) ** 3
    )


# DrCIF
def f_cubic_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.891346 + 0.010175 * (x1 - 1.458503) - 0.001145 * (x1 - 4.559232) ** 3
    )


def sigma_quadratic_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.046291 - 0.192197 * (x1 + 0.421337) + 0.008213 * (x1 + 11.075024) ** 2
    )


# EE
def f_sinusoidal_ee_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        3.679372 - 0.078029 * (x1 + 27.163093) - 0.791569 * np.sin(x1 + 0.249687)
    )


def sigma_cubic_ee_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        14.351948 + 0.350017 * (x1 - 55.021346) - 0.008466 * (x1 - 1.546773) ** 3
    )


# FreshPRINCE
def f_cubic_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.598212 + 0.090616 * (x1 + 1.838934) - 0.002273 * (x1 - 3.434601) ** 3
    )


def sigma_quadratic_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -1.028964 - 0.231054 * (x1 + 19.464579) + 0.008203 * (x1 + 13.189041) ** 2
    )


# HC1
def f_sinusoidal_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -3.796373 - 0.036011 * (x1 - 173.306368) - 0.983149 * np.sin(x1 + 0.232909)
    )


def sigma_cubic_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -11.990515 + 0.188011 * (x1 + 40.635945) - 0.006555 * (x1 - 0.439561) ** 3
    )


# HC2
def f_cosine_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        3.14459 - 0.032208 * (x1 + 16.118847) + 1.046249 * np.cos(x1 + 8.065588)
    )


def sigma_cubic_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.902125 + 0.208744 * (x1 + 3.312879) - 0.00699 * (x1 - 0.171413) ** 3
    )


# Hydra-MR
def f_sinusoidal_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.257881 - 0.109071 * (x1 - 21.809879) - 1.725171 * np.sin(x1 + 0.269034)
    )


def sigma_quadratic_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -6.13329 - 0.131009 * (x1 - 4.839777) + 0.049945 * (x1 + 1.576739) ** 2
    )


# Hydra
def f_cosine_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.666259 - 0.051857 * (x1 + 0.017538) + 1.500695 * np.cos(x1 + 8.124965)
    )


def sigma_cubic_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.755217 + 0.232768 * (x1 - 2.188701) - 0.007811 * (x1 - 0.280014) ** 3
    )


# InceptionT
def f_cubic_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -7.716966 + 0.094367 * (x1 + 101.497791) - 0.004167 * (x1 + 0.109162) ** 3
    )


def sigma_quadratic_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -5.730842 - 0.084045 * (x1 - 15.819096) + 0.022673 * (x1 + 2.931101) ** 2
    )


# Mini-R
def f_exponential_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.965902 - 0.006666 * (x1 + 0.35949) - 0.002899 * np.exp(x1 + 0.853629)
    )


def sigma_exponential_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.062777 - 0.006631 * (x1 - 13.600128) - 0.047995 * np.exp(x1 - 4.011454)
    )


# MrSQM
def f_quadratic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(0.96716 - 0.101148 * (x1 + 2.210559) + 0.045723 * (x1 + 1.915524) ** 2)


def sigma_cubic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.010538 - 0.097529 * (x1 + 1.789462) + 0.001355 * (x1 + 1.900384) ** 3
    )


# Multi-R
def f_cosine_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.652187 - 0.024368 * (x1 - 0.459955) + 1.118281 * np.cos(x1 + 8.160833)
    )


def sigma_cubic_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -7.621192 + 0.292581 * (x1 + 11.812293) - 0.009577 * (x1 - 0.124038) ** 3
    )


# PF
def f_exponential_pf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(1.0267 - 0.148475 * (x1 + 0.523744) + 0.003921 * np.exp(x1 - 0.103493))


def sigma_cubic_pf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        1.184282 + 0.106983 * (x1 - 50.924763) - 0.00131 * (x1 - 3.839002) ** 3
    )


# RDST
def f_cubic_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.374679 + 0.501382 * (x1 - 12.630988) - 6.3e-05 * (x1 - 48.022867) ** 3
    )


def sigma_cubic_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -1.183852 + 0.403323 * (x1 - 5.695905) - 0.012445 * (x1 + 0.456515) ** 3
    )


# RISE
def f_cubic_rise_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.18699 + 0.032061 * (x1 + 25.982174) - 0.002982 * (x1 - 1.547126) ** 3
    )


def sigma_cubic_rise_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -4.628189 - 0.121223 * (x1 - 2.466433) + 0.00091 * (x1 + 4.827967) ** 3
    )


# RIST
def f_exponential_rist_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -1.347732 - 1.097222 * (x1 + 3.302676) + 0.00808 * np.exp(x1 + 9.46592)
    )


def sigma_sinusoidal_rist_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        0.887236 - 0.305171 * (x1 + 22.398818) + 2.218922 * np.sin(x1 - 0.169696)
    )


# ROCKET
def f_exponential_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.911435 - 0.098079 * (x1 - 5.397661) - 0.026838 * np.exp(x1 - 1.634123)
    )


def sigma_cubic_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -17.441336 + 0.055372 * (x1 + 245.024866) - 0.002027 * (x1 + 0.537528) ** 3
    )


# RSF
def f_cosine_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.971774 + 0.048461 * (x1 + 14.645142) - 0.101842 * np.cos(x1 + 0.681274)
    )


def sigma_cubic_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.645493 + 0.667401 * (x1 + 0.997882) - 0.020849 * (x1 + 0.399996) ** 3
    )


# RSTSF
def f_cubic_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.131246 + 0.048524 * (x1 - 26.320879) - 0.001569 * (x1 - 4.101215) ** 3
    )


def sigma_quadratic_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.704639 - 0.146014 * (x1 + 6.498178) + 0.012009 * (x1 + 6.67758) ** 2
    )


# ResNet
def f_sinusoidal_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        2.862551 + 0.047341 * (x1 - 13.330246) - 0.450571 * np.sin(x1 + 0.699709)
    )


def sigma_cubic_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -1.456432 + 8.780995 * (x1 - 2.513521) - 0.241182 * (x1 - 0.753677) ** 3
    )


# STC
def f_sinusoidal_stc_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        1.230375 - 0.0186 * (x1 - 55.579543) - 0.925654 * np.sin(x1 + 0.229553)
    )


def sigma_cubic_stc_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        0.877268 + 0.175156 * (x1 - 27.861698) - 0.006864 * (x1 - 0.17633) ** 3
    )


# STSF
def f_cubic_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.643128 - 0.006396 * (x1 - 14.728583) - 0.001748 * (x1 - 3.012455) ** 3
    )


def sigma_cubic_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.210135 - 0.097623 * (x1 + 10.849915) + 0.000598 * (x1 + 6.80661) ** 3
    )


# ShapeDTW
def f_sinusoidal_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -1.298306 - 0.114211 * (x1 - 15.343923) - 0.152522 * np.sin(x1 - 1.762558)
    )


def sigma_quadratic_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -0.907174 - 0.21036 * (x1 + 18.523593) + 0.017924 * (x1 + 4.482362) ** 2
    )


# Signatures
def f_cosine_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -5.703277 - 0.119271 * (x1 - 52.672128) + 0.304926 * np.cos(x1 - 0.684961)
    )


def sigma_cubic_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -10.013202 + 1.738331 * (x1 - 39.293312) - 3.8e-05 * (x1 - 125.033329) ** 3
    )


# TDE
def f_cubic_tde_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -160.118537 + 11.584089 * (x1 - 71.625492) - 5.8e-05 * (x1 - 257.98172) ** 3
    )


def sigma_cosine_tde_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -6.849197 - 0.06437 * (x1 - 42.501307) - 0.083439 * np.cos(x1 + 0.935747)
    )


# TS-CHIEF
def f_cosine_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.040078 - 0.069501 * (x1 - 38.65634) + 1.557393 * np.cos(x1 + 8.079209)
    )


def sigma_cubic_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        21.848163 + 0.18742 * (x1 - 140.245545) - 0.006344 * (x1 - 0.539176) ** 3
    )


# TSF
def f_sinusoidal_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -3.232277 - 0.110677 * (x1 - 36.210796) + 0.217935 * np.sin(x1 - 5.052419)
    )


def sigma_cubic_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -9.002519 + 1.10917 * (x1 - 21.477508) - 6.8e-05 * (x1 - 75.073467) ** 3
    )


# TSFresh
def f_cubic_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -0.806271 - 0.119361 * (x1 - 23.261874) + 0.0088 * (x1 + 0.384648) ** 3
    )


def sigma_cubic_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -2.156059 + 2.358569 * (x1 - 0.794105) - 0.131037 * (x1 + 1.041635) ** 3
    )


# WEASEL-D
def f_cubic_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -27.111943 + 6.361531 * (x1 - 43.464658) - 0.0001 * (x1 - 145.134837) ** 3
    )


def sigma_quadratic_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -3.665889 - 0.042993 * (x1 + 12.97554) + 0.005916 * (x1 + 2.061859) ** 2
    )


# WEASEL
def f_quadratic_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        0.680878 - 0.073143 * (x1 - 3.513466) + 0.012544 * (x1 - 1.523125) ** 2
    )


def sigma_cubic_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -2.440648 + 0.113076 * (x1 - 12.639588) - 0.000879 * (x1 - 3.811396) ** 3
    )


# cBOSS
def f_cubic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the true mean function f(x) for 1D input."""
    x1 = X[:, 0]
    return expit(
        -150.075644 + 9.913785 * (x1 - 59.205162) - 6.5e-05 * (x1 - 224.771216) ** 3
    )


def sigma_quadratic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    """Compute the heteroscedastic noise σ(x) for 1D input."""
    x1 = X[:, 0]
    return softplus(
        -2.514629 - 0.125268 * (x1 + 15.10036) + 0.008365 * (x1 + 4.549561) ** 2
    )


### 2 FEATURES ###
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
        -4.999376 - 0.106637 * (x1 + 1.131506) + 0.002686 * (x2 - 9.444566) ** 2
    )


FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "f_exponential__1nn_dtw_1_feat": f_exponential__1nn_dtw_1_feat,
    "f_exponential_arsenal_1_feat": f_exponential_arsenal_1_feat,
    "f_cubic_boss_1_feat": f_cubic_boss_1_feat,
    "f_cosine_cif_1_feat": f_cosine_cif_1_feat,
    "f_cosine_cnn_1_feat": f_cosine_cnn_1_feat,
    "f_cubic_catch22_1_feat": f_cubic_catch22_1_feat,
    "f_cubic_drcif_1_feat": f_cubic_drcif_1_feat,
    "f_sinusoidal_ee_1_feat": f_sinusoidal_ee_1_feat,
    "f_cubic_freshprince_1_feat": f_cubic_freshprince_1_feat,
    "f_sinusoidal_hc1_1_feat": f_sinusoidal_hc1_1_feat,
    "f_cosine_hc2_1_feat": f_cosine_hc2_1_feat,
    "f_sinusoidal_hydra_mr_1_feat": f_sinusoidal_hydra_mr_1_feat,
    "f_cosine_hydra_1_feat": f_cosine_hydra_1_feat,
    "f_cubic_inceptiont_1_feat": f_cubic_inceptiont_1_feat,
    "f_exponential_mini_r_1_feat": f_exponential_mini_r_1_feat,
    "f_quadratic_mrsqm_1_feat": f_quadratic_mrsqm_1_feat,
    "f_cosine_multi_r_1_feat": f_cosine_multi_r_1_feat,
    "f_exponential_pf_1_feat": f_exponential_pf_1_feat,
    "f_cubic_rdst_1_feat": f_cubic_rdst_1_feat,
    "f_cubic_rise_1_feat": f_cubic_rise_1_feat,
    "f_exponential_rist_1_feat": f_exponential_rist_1_feat,
    "f_exponential_rocket_1_feat": f_exponential_rocket_1_feat,
    "f_cosine_rsf_1_feat": f_cosine_rsf_1_feat,
    "f_cubic_rstsf_1_feat": f_cubic_rstsf_1_feat,
    "f_sinusoidal_resnet_1_feat": f_sinusoidal_resnet_1_feat,
    "f_sinusoidal_stc_1_feat": f_sinusoidal_stc_1_feat,
    "f_cubic_stsf_1_feat": f_cubic_stsf_1_feat,
    "f_sinusoidal_shapedtw_1_feat": f_sinusoidal_shapedtw_1_feat,
    "f_cosine_signatures_1_feat": f_cosine_signatures_1_feat,
    "f_cubic_tde_1_feat": f_cubic_tde_1_feat,
    "f_cosine_ts_chief_1_feat": f_cosine_ts_chief_1_feat,
    "f_sinusoidal_tsf_1_feat": f_sinusoidal_tsf_1_feat,
    "f_cubic_tsfresh_1_feat": f_cubic_tsfresh_1_feat,
    "f_cubic_weasel_d_1_feat": f_cubic_weasel_d_1_feat,
    "f_quadratic_weasel_1_feat": f_quadratic_weasel_1_feat,
    "f_cubic_cboss_1_feat": f_cubic_cboss_1_feat,
    # 2 Features
    "f_quadratic_hc2_2_feat": f_quadratic_hc2_2_feat,
}

SIGMAS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigma_cubic__1nn_dtw_1_feat": sigma_cubic__1nn_dtw_1_feat,
    "sigma_cubic_arsenal_1_feat": sigma_cubic_arsenal_1_feat,
    "sigma_quadratic_boss_1_feat": sigma_quadratic_boss_1_feat,
    "sigma_cubic_cif_1_feat": sigma_cubic_cif_1_feat,
    "sigma_cubic_cnn_1_feat": sigma_cubic_cnn_1_feat,
    "sigma_cubic_catch22_1_feat": sigma_cubic_catch22_1_feat,
    "sigma_quadratic_drcif_1_feat": sigma_quadratic_drcif_1_feat,
    "sigma_cubic_ee_1_feat": sigma_cubic_ee_1_feat,
    "sigma_quadratic_freshprince_1_feat": sigma_quadratic_freshprince_1_feat,
    "sigma_cubic_hc1_1_feat": sigma_cubic_hc1_1_feat,
    "sigma_cubic_hc2_1_feat": sigma_cubic_hc2_1_feat,
    "sigma_quadratic_hydra_mr_1_feat": sigma_quadratic_hydra_mr_1_feat,
    "sigma_cubic_hydra_1_feat": sigma_cubic_hydra_1_feat,
    "sigma_quadratic_inceptiont_1_feat": sigma_quadratic_inceptiont_1_feat,
    "sigma_exponential_mini_r_1_feat": sigma_exponential_mini_r_1_feat,
    "sigma_cubic_mrsqm_1_feat": sigma_cubic_mrsqm_1_feat,
    "sigma_cubic_multi_r_1_feat": sigma_cubic_multi_r_1_feat,
    "sigma_cubic_pf_1_feat": sigma_cubic_pf_1_feat,
    "sigma_cubic_rdst_1_feat": sigma_cubic_rdst_1_feat,
    "sigma_cubic_rise_1_feat": sigma_cubic_rise_1_feat,
    "sigma_sinusoidal_rist_1_feat": sigma_sinusoidal_rist_1_feat,
    "sigma_cubic_rocket_1_feat": sigma_cubic_rocket_1_feat,
    "sigma_cubic_rsf_1_feat": sigma_cubic_rsf_1_feat,
    "sigma_quadratic_rstsf_1_feat": sigma_quadratic_rstsf_1_feat,
    "sigma_cubic_resnet_1_feat": sigma_cubic_resnet_1_feat,
    "sigma_cubic_stc_1_feat": sigma_cubic_stc_1_feat,
    "sigma_cubic_stsf_1_feat": sigma_cubic_stsf_1_feat,
    "sigma_quadratic_shapedtw_1_feat": sigma_quadratic_shapedtw_1_feat,
    "sigma_cubic_signatures_1_feat": sigma_cubic_signatures_1_feat,
    "sigma_cosine_tde_1_feat": sigma_cosine_tde_1_feat,
    "sigma_cubic_ts_chief_1_feat": sigma_cubic_ts_chief_1_feat,
    "sigma_cubic_tsf_1_feat": sigma_cubic_tsf_1_feat,
    "sigma_cubic_tsfresh_1_feat": sigma_cubic_tsfresh_1_feat,
    "sigma_quadratic_weasel_d_1_feat": sigma_quadratic_weasel_d_1_feat,
    "sigma_cubic_weasel_1_feat": sigma_cubic_weasel_1_feat,
    "sigma_quadratic_cboss_1_feat": sigma_quadratic_cboss_1_feat,
    # 2 Features
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
        self.rng_train = np.random.default_rng(seed=int(job["seed"]))
        # independent RNG for test data
        self.rng_test = np.random.default_rng(seed=int(job["seed"]) + 1)

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
            sampler = qmc.LatinHypercube(d=n_features, seed=self.rng_train)
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
                s1 = qmc.LatinHypercube(d=1, seed=self.rng_train)
                s2 = qmc.LatinHypercube(d=1, seed=self.rng_train)
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
                    s1 = qmc.LatinHypercube(d=1, seed=self.rng_train)
                    s2 = qmc.LatinHypercube(d=1, seed=self.rng_train)
                    part1 = qmc.scale(s1.random(n1), [a], [b]).ravel()
                    part2 = qmc.scale(s2.random(n2), [c], [d]).ravel()
                    base_axis = np.concatenate([part1, part2])
                    axes.append(base_axis[self.rng_train.permutation(n_instances)])
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
        noise = self.rng_train.normal(0.0, sigma)
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
        y_clean = fn(X)  # (N,) or (N,1)
        sigma = sigma_fn(X)  # (N,) or (N,1)
        noise = self.rng_test.normal(0.0, sigma)
        y = y_clean + noise

        return {"X": X, "y": y, "sigma": sigma, "y_clean": y_clean}

    def _probe_dim(self, f: Callable[[np.ndarray], np.ndarray]) -> int:
        """Infer input dimensionality D of `f` by probing shapes (N, D) for D=1..9999.

        Args:
            f: Callable mapping (N, D) arrays to targets.

        Returns:
            Detected input dimension D.
        """
        for d in range(1, 9999):
            try:
                _ = f(np.zeros((1, d)))
                return d
            except Exception:
                continue
