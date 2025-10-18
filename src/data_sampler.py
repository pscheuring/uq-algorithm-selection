from collections.abc import Callable
from typing import Callable, Dict, Union

import numpy as np
from scipy.special import expit, softplus
from scipy.stats import qmc


# 1NN-DTW
def f_exponential_1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.120265 - 0.591617 * (x1 + 1.656651) + 0.588102 * np.exp(x1 - 1.114355)
    )


def sigma_cosine_1nn_dtw_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -5.895627 - 0.238146 * (x1 - 8.141458) + 0.575536 * np.cos(x1 - 2.562086)
    )


# Arsenal
def f_cubic_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        4.892204 + 0.009822 * (x1 - 308.642698) - 0.139292 * (x1 + 0.941515) ** 3
    )


def sigma_cosine_arsenal_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.861460 - 0.159958 * (x1 + 6.558437) - 0.307698 * np.cos(x1 + 1.689044)
    )


# BOSS
def f_quadratic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        0.445399 + 0.191964 * (x1 + 0.614573) + 0.442974 * (x1 + 0.015217) ** 2
    )


def sigma_quadratic_boss_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.972103 - 0.130375 * (x1 + 7.911297) + 0.019372 * (x1 - 0.208795) ** 2
    )


# CIF
def f_sinusoidal_cif_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -0.028261 - 0.459734 * (x1 - 2.580903) + 0.330796 * np.sin(x1 - 0.904447)
    )


def sigma_sinusoidal_cif_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -4.146880 - 0.063335 * (x1 - 3.362816) + 0.403485 * np.sin(x1 - 1.501530)
    )


# CNN
def f_cosine_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -0.613924 + 0.698288 * (x1 + 3.034605) + 0.921704 * np.cos(x1 + 1.261144)
    )


def sigma_cosine_cnn_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.763116 - 0.487912 * (x1 + 1.995168) - 0.675727 * np.cos(x1 + 1.047064)
    )


# Catch22
def f_cosine_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.199424 + 0.182007 * (x1 - 2.616731) + 0.203167 * np.cos(x1 + 0.037591)
    )


def sigma_cubic_catch22_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -22.448169 + 1.139947 * (x1 + 15.841750) - 0.327444 * (x1 - 0.165658) ** 3
    )


# DrCIF
def f_sinusoidal_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.879972 - 0.482604 * (x1 + 1.122380) + 0.484172 * np.sin(x1 - 0.805897)
    )


def sigma_quadratic_drcif_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -10.001079 - 0.654516 * (x1 - 6.975360) + 0.073913 * (x1 + 4.219171) ** 2
    )


# EE
def f_cosine_ee_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.592441 - 0.489938 * (x1 + 2.097928) + 1.072883 * np.cos(x1 - 1.050211)
    )


def sigma_cosine_ee_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        2.733952 + 0.487740 * (x1 - 13.137751) + 1.094978 * np.cos(x1 + 2.363619)
    )


# FreshPRINCE
def f_sinusoidal_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -3.459745 - 0.511589 * (x1 - 9.483761) + 0.824131 * np.sin(x1 - 0.755529)
    )


def sigma_quadratic_freshprince_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -6.440138 + 0.240042 * (x1 + 8.613122) + 0.073829 * (x1 - 1.923748) ** 2
    )


# HC1
def f_cosine_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.513277 - 0.108548 * (x1 + 3.066431) + 0.467545 * np.cos(x1 + 4.520223)
    )


def sigma_sinusoidal_hc1_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -8.527643 - 0.844332 * (x1 - 5.187394) + 1.539118 * np.sin(x1 - 0.164551)
    )


# HC2
def f_cosine_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.284111 + 0.300723 * (x1 + 3.293876) + 1.296269 * np.cos(x1 + 3.027899)
    )


def sigma_exponential_hc2_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        1.467056 - 0.134861 * (x1 + 42.180005) + 104.721256 * np.exp(x1 - 103.792825)
    )


# Hydra-MR
def f_sinusoidal_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        3.579104 - 0.072378 * (x1 + 23.238464) + 0.778236 * np.sin(x1 - 1.421729)
    )


def sigma_cosine_hydra_mr_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        67.808367 + 0.323940 * (x1 - 221.623459) + 0.680687 * np.cos(x1 + 1.984960)
    )


# Hydra
def f_exponential_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.374426 - 0.883025 * (x1 + 1.277315) + 0.368527 * np.exp(x1 + 0.633187)
    )


def sigma_quadratic_hydra_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -3.365565 - 0.084115 * (x1 + 8.669651) + 0.014603 * (x1 + 0.751103) ** 2
    )


# InceptionT
def f_sinusoidal_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.709397 + 0.139420 * (x1 + 2.199950) + 0.951247 * np.sin(x1 - 1.560090)
    )


def sigma_cosine_inceptiont_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.317912 + 0.225215 * (x1 - 7.233668) + 0.605986 * np.cos(x1 + 2.166695)
    )


# Mini-R
def f_sinusoidal_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.798051 + 0.261173 * (x1 + 0.934584) + 1.107410 * np.sin(x1 - 1.728273)
    )


def sigma_quadratic_mini_r_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -4.659254 - 0.139596 * (x1 - 3.466171) + 0.011010 * (x1 + 1.120437) ** 2
    )


# MrSQM
def f_quadratic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        0.590175 + 0.528483 * (x1 + 0.499571) + 0.411510 * (x1 - 0.372317) ** 2
    )


def sigma_cubic_mrsqm_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -9.268910 - 0.292502 * (x1 - 17.381137) + 0.036550 * (x1 + 0.633502) ** 3
    )


# Multi-R
def f_sinusoidal_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.545687 - 0.107855 * (x1 - 2.666674) + 0.710534 * np.sin(x1 - 1.399250)
    )


def sigma_sinusoidal_multi_r_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.275641 + 0.235255 * (x1 - 7.417669) + 0.562124 * np.sin(x1 - 2.577143)
    )


# PF
def f_quadratic_pf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.177192 - 0.134291 * (x1 + 2.350196) + 0.109982 * (x1 - 0.602218) ** 2
    )


def sigma_cosine_pf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.056647 - 0.125263 * (x1 + 14.146840) - 0.556884 * np.cos(x1 + 0.583031)
    )


# RDST
def f_cubic_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -203.046362 + 1.110719 * (x1 + 206.182912) - 4.832327 * (x1 - 0.260362) ** 3
    )


def sigma_sinusoidal_rdst_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -5.895310 + 0.139832 * (x1 + 12.625951) - 0.515828 * np.sin(x1 + 0.696506)
    )


# RISE
def f_cubic_rise_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -27.159027 + 0.096182 * (x1 + 289.088960) - 0.080502 * (x1 - 0.515710) ** 3
    )


def sigma_sinusoidal_rise_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -4.005273 + 0.005335 * (x1 + 18.243475) - 0.424416 * np.sin(x1 - 23.959892)
    )


# RIST
def f_sinusoidal_rist_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.397954 - 0.233607 * (x1 - 1.872494) - 0.985998 * np.sin(x1 + 2.049964)
    )


def sigma_cosine_rist_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        3.309803 + 0.765083 * (x1 - 9.327624) + 1.423245 * np.cos(x1 + 1.714843)
    )


# ROCKET
def f_cubic_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        -13.513560 + 0.048740 * (x1 + 316.761869) - 0.138214 * (x1 + 0.899301) ** 3
    )


def sigma_cubic_rocket_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -7.385054 + 0.166115 * (x1 + 21.136979) - 0.054731 * (x1 + 0.179175) ** 3
    )


# RSF
def f_quadratic_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.623405 + 0.302210 * (x1 + 1.085313) - 0.064485 * (x1 + 1.398748) ** 2
    )


def sigma_cubic_rsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        9.737397 + 1.208458 * (x1 - 11.867914) - 0.360238 * (x1 - 0.186952) ** 3
    )


# RSTSF
def f_cosine_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        4.978427 - 0.479888 * (x1 + 7.451978) + 0.630222 * np.cos(x1 + 3.899661)
    )


def sigma_cosine_rstsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -0.973783 + 0.117289 * (x1 - 24.280646) + 0.439357 * np.cos(x1 + 2.889615)
    )


# ResNet
def f_cosine_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.790821 + 0.387048 * (x1 - 1.997260) + 0.392089 * np.cos(x1 + 1.125142)
    )


def sigma_cosine_resnet_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -8.755075 - 29.429915 * (x1 + 0.101980) - 65.287971 * np.cos(x1 + 13.844889)
    )


# STC
def f_sinusoidal_stc_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.118110 - 0.309074 * (x1 - 1.397459) + 0.837192 * np.sin(x1 - 0.970293)
    )


def sigma_cosine_stc_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -3.099241 + 0.030613 * (x1 - 16.469512) + 0.469656 * np.cos(x1 + 3.336999)
    )


# STSF
def f_cubic_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.668263 - 0.019188 * (x1 + 48.518535) - 0.047203 * (x1 - 1.004110) ** 3
    )


def sigma_sinusoidal_stsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -4.351962 + 0.082878 * (x1 + 6.863329) - 0.405409 * np.sin(x1 + 1.287990)
    )


# ShapeDTW
def f_cosine_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        7.013002 - 0.497616 * (x1 + 13.192100) - 0.204860 * np.cos(x1 + 1.294217)
    )


def sigma_cosine_shapedtw_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -30.252453 - 0.308708 * (x1 - 84.909923) + 0.470622 * np.cos(x1 + 3.735122)
    )


# Signatures
def f_cosine_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        3.717227 - 0.675096 * (x1 + 4.664440) + 0.366628 * np.cos(x1 - 1.705026)
    )


def sigma_cubic_signatures_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -16.783518 + 2.029947 * (x1 + 0.185427) - 0.002469 * (x1 - 17.000710) ** 3
    )


# TDE
def f_cosine_tde_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.156934 + 0.333877 * (x1 + 2.481203) + 1.306953 * np.cos(x1 + 2.986082)
    )


def sigma_exponential_tde_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -2.885686 - 0.169406 * (x1 + 6.985851) + 104.722649 * np.exp(x1 - 103.794081)
    )


# TS-CHIEF
def f_exponential_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.144413 + 9.169916 * (x1 + 2.044067) - 0.522003 * np.exp(x1 + 2.277841)
    )


def sigma_quadratic_ts_chief_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        5.229433 + 1.137278 * (x1 - 9.171682) + 0.321213 * (x1 - 1.304490) ** 2
    )


# TSF
def f_quadratic_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        0.111493 - 0.463924 * (x1 - 0.864954) + 0.112234 * (x1 + 0.541899) ** 2
    )


def sigma_sinusoidal_tsf_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -3.120486 - 0.114673 * (x1 + 6.775996) - 0.458296 * np.sin(x1 + 1.652590)
    )


# TSFresh
def f_exponential_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.360527 - 0.420910 * (x1 + 1.615534) + 0.215117 * np.exp(x1 - 0.489828)
    )


def sigma_quadratic_tsfresh_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -9.737740 - 1.116599 * (x1 - 4.009632) + 0.293304 * (x1 + 2.116193) ** 2
    )


# WEASEL-D
def f_sinusoidal_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.076360 + 0.022062 * (x1 - 4.108363) + 1.061914 * np.sin(x1 - 1.363218)
    )


def sigma_sinusoidal_weasel_d_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -5.644733 + 0.189935 * (x1 + 8.532192) - 0.537640 * np.sin(x1 + 0.751025)
    )


# WEASEL
def f_exponential_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        2.172744 - 0.653566 * (x1 + 2.226568) + 0.128421 * np.exp(x1 + 0.516852)
    )


def sigma_quadratic_weasel_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -4.260620 - 0.086199 * (x1 - 3.113799) + 0.106754 * (x1 + 1.122685) ** 2
    )


# cBOSS
def f_quadratic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return expit(
        1.191922 - 0.219652 * (x1 + 1.212123) + 4.387714 * (x1 + 1.691059) ** 2
    )


def sigma_cubic_cboss_1_feat(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return softplus(
        -9.094248 + 0.594885 * (x1 + 7.673351) - 0.182437 * (x1 - 0.300354) ** 3
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


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "f_exponential_1nn_dtw_1_feat": f_exponential_1nn_dtw_1_feat,
    "f_cubic_arsenal_1_feat": f_cubic_arsenal_1_feat,
    "f_quadratic_boss_1_feat": f_quadratic_boss_1_feat,
    "f_sinusoidal_cif_1_feat": f_sinusoidal_cif_1_feat,
    "f_cosine_cnn_1_feat": f_cosine_cnn_1_feat,
    "f_cosine_catch22_1_feat": f_cosine_catch22_1_feat,
    "f_sinusoidal_drcif_1_feat": f_sinusoidal_drcif_1_feat,
    "f_cosine_ee_1_feat": f_cosine_ee_1_feat,
    "f_sinusoidal_freshprince_1_feat": f_sinusoidal_freshprince_1_feat,
    "f_cosine_hc1_1_feat": f_cosine_hc1_1_feat,
    "f_cosine_hc2_1_feat": f_cosine_hc2_1_feat,
    "f_sinusoidal_hydra_mr_1_feat": f_sinusoidal_hydra_mr_1_feat,
    "f_exponential_hydra_1_feat": f_exponential_hydra_1_feat,
    "f_sinusoidal_inceptiont_1_feat": f_sinusoidal_inceptiont_1_feat,
    "f_sinusoidal_mini_r_1_feat": f_sinusoidal_mini_r_1_feat,
    "f_quadratic_mrsqm_1_feat": f_quadratic_mrsqm_1_feat,
    "f_sinusoidal_multi_r_1_feat": f_sinusoidal_multi_r_1_feat,
    "f_quadratic_pf_1_feat": f_quadratic_pf_1_feat,
    "f_cubic_rdst_1_feat": f_cubic_rdst_1_feat,
    "f_cubic_rise_1_feat": f_cubic_rise_1_feat,
    "f_sinusoidal_rist_1_feat": f_sinusoidal_rist_1_feat,
    "f_cubic_rocket_1_feat": f_cubic_rocket_1_feat,
    "f_quadratic_rsf_1_feat": f_quadratic_rsf_1_feat,
    "f_cosine_rstsf_1_feat": f_cosine_rstsf_1_feat,
    "f_cosine_resnet_1_feat": f_cosine_resnet_1_feat,
    "f_sinusoidal_stc_1_feat": f_sinusoidal_stc_1_feat,
    "f_cubic_stsf_1_feat": f_cubic_stsf_1_feat,
    "f_cosine_shapedtw_1_feat": f_cosine_shapedtw_1_feat,
    "f_cosine_signatures_1_feat": f_cosine_signatures_1_feat,
    "f_cosine_tde_1_feat": f_cosine_tde_1_feat,
    "f_exponential_ts_chief_1_feat": f_exponential_ts_chief_1_feat,
    "f_quadratic_tsf_1_feat": f_quadratic_tsf_1_feat,
    "f_exponential_tsfresh_1_feat": f_exponential_tsfresh_1_feat,
    "f_sinusoidal_weasel_d_1_feat": f_sinusoidal_weasel_d_1_feat,
    "f_exponential_weasel_1_feat": f_exponential_weasel_1_feat,
    "f_quadratic_cboss_1_feat": f_quadratic_cboss_1_feat,
    "f_quadratic_rocket_2_feat": f_quadratic_rocket_2_feat,
}


SIGMAS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigma_cosine_1nn_dtw_1_feat": sigma_cosine_1nn_dtw_1_feat,
    "sigma_cosine_arsenal_1_feat": sigma_cosine_arsenal_1_feat,
    "sigma_quadratic_boss_1_feat": sigma_quadratic_boss_1_feat,
    "sigma_sinusoidal_cif_1_feat": sigma_sinusoidal_cif_1_feat,
    "sigma_cosine_cnn_1_feat": sigma_cosine_cnn_1_feat,
    "sigma_cubic_catch22_1_feat": sigma_cubic_catch22_1_feat,
    "sigma_quadratic_drcif_1_feat": sigma_quadratic_drcif_1_feat,
    "sigma_cosine_ee_1_feat": sigma_cosine_ee_1_feat,
    "sigma_quadratic_freshprince_1_feat": sigma_quadratic_freshprince_1_feat,
    "sigma_sinusoidal_hc1_1_feat": sigma_sinusoidal_hc1_1_feat,
    "sigma_exponential_hc2_1_feat": sigma_exponential_hc2_1_feat,
    "sigma_cosine_hydra_mr_1_feat": sigma_cosine_hydra_mr_1_feat,
    "sigma_quadratic_hydra_1_feat": sigma_quadratic_hydra_1_feat,
    "sigma_cosine_inceptiont_1_feat": sigma_cosine_inceptiont_1_feat,
    "sigma_quadratic_mini_r_1_feat": sigma_quadratic_mini_r_1_feat,
    "sigma_cubic_mrsqm_1_feat": sigma_cubic_mrsqm_1_feat,
    "sigma_sinusoidal_multi_r_1_feat": sigma_sinusoidal_multi_r_1_feat,
    "sigma_cosine_pf_1_feat": sigma_cosine_pf_1_feat,
    "sigma_sinusoidal_rdst_1_feat": sigma_sinusoidal_rdst_1_feat,
    "sigma_sinusoidal_rise_1_feat": sigma_sinusoidal_rise_1_feat,
    "sigma_cosine_rist_1_feat": sigma_cosine_rist_1_feat,
    "sigma_cubic_rocket_1_feat": sigma_cubic_rocket_1_feat,
    "sigma_cubic_rsf_1_feat": sigma_cubic_rsf_1_feat,
    "sigma_cosine_rstsf_1_feat": sigma_cosine_rstsf_1_feat,
    "sigma_cosine_resnet_1_feat": sigma_cosine_resnet_1_feat,
    "sigma_cosine_stc_1_feat": sigma_cosine_stc_1_feat,
    "sigma_sinusoidal_stsf_1_feat": sigma_sinusoidal_stsf_1_feat,
    "sigma_cosine_shapedtw_1_feat": sigma_cosine_shapedtw_1_feat,
    "sigma_cubic_signatures_1_feat": sigma_cubic_signatures_1_feat,
    "sigma_exponential_tde_1_feat": sigma_exponential_tde_1_feat,
    "sigma_quadratic_ts_chief_1_feat": sigma_quadratic_ts_chief_1_feat,
    "sigma_sinusoidal_tsf_1_feat": sigma_sinusoidal_tsf_1_feat,
    "sigma_quadratic_tsfresh_1_feat": sigma_quadratic_tsfresh_1_feat,
    "sigma_sinusoidal_weasel_d_1_feat": sigma_sinusoidal_weasel_d_1_feat,
    "sigma_quadratic_weasel_1_feat": sigma_quadratic_weasel_1_feat,
    "sigma_cubic_cboss_1_feat": sigma_cubic_cboss_1_feat,
    "sigma_cosine_rocket_2_feat": sigma_cosine_rocket_2_feat,
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
        test_points (optional, list[float]): Extra 1D points; broadcast to all dims if D>1.

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
        functions: dict[str, Callable[[np.ndarray], np.ndarray]],
        sigmas: dict[str, Callable[[np.ndarray], np.ndarray]],
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
            else np.repeat(X_unique, repeats=n_repeats, axis=0)
        )

        # Targets + heteroscedastic noise
        y_clean = fn(X)  # (N,) or (N,1)
        sigma = sigma_fn(X)  # (N,) or (N,1)
        noise = self.rng.normal(0.0, sigma)
        y = y_clean + noise

        return {
            "X": X,
            "y": y,
            "y_clean": y_clean,
            "sigma": sigma,
            "n_features": np.int64(n_features),
        }

    def sample_test_data(self) -> dict[str, np.ndarray]:
        """Sample test data on a uniform Cartesian grid spanning `test_interval`.

        Builds a grid with `test_grid_length` points per dimension.
        Optionally appends 1D `test_points` (broadcast across dims when D>1).
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

        axis = np.linspace(lo, hi, grid_length, dtype=np.float64)
        mesh = np.meshgrid(*([axis] * n_features), indexing="xy")
        X_grid = np.stack(mesh, axis=-1).reshape(-1, n_features)

        X_list: list[np.ndarray] = [X_grid]

        test_points = self.job.get("test_points", None)
        if test_points is not None:
            # flat list of scalars → use as-is for D==1, else broadcast across dims
            if all(not isinstance(t, (list, tuple, np.ndarray)) for t in test_points):
                vals = np.asarray(test_points, dtype=np.float64).reshape(-1, 1)
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
