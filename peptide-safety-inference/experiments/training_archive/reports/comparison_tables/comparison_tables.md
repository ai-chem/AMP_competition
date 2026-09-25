# Model comparison tables

Locked-test metrics use seed `20260925` and were scored once per protocol version.
Production weights are a post-selection refit on all suitable labels;
they are **not** re-evaluated on the locked test.

## HC50 — locked test (cluster-disjoint)

| version | description | pearson_r | spearman_rho | r2 | mae | rmse | roc_auc | pr_auc | mcc | right_censored_consistency | coverage_90 | n_test_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_svr | SVR physchem+ESM-2 | 0.312 | 0.340 | 0.065 | 1.230 | 1.692 | 0.796 | 0.788 | 0.364 | 0.415 | 0.589 | 338.000 |
| v3_svr_tobit | SVR + Tobit PCA blend | 0.313 | 0.357 | 0.056 | 1.209 | 1.699 | 0.811 | 0.807 | 0.460 | 0.483 | 0.941 | 338.000 |
| v4_boost | CatBoost + ExtraTrees | 0.317 | 0.320 | 0.100 | 1.168 | 1.659 | 0.801 | 0.798 | 0.437 | 0.319 | 0.932 | 338.000 |
| v5_constrained | CB+ET+Tobit (RC constrained) | 0.302 | 0.320 | 0.082 | 1.170 | 1.676 | 0.803 | 0.797 | 0.431 | 0.407 | 0.932 | 338.000 |
| v7_esm_ft | ESM-2 35M fine-tune (selected) | 0.317 | 0.383 | -0.138 | 1.309 | 1.866 | 0.837 | 0.829 | 0.545 | 0.807 | 0.926 | 338.000 |

## HC50 — selection / OOF

| stage | model | features | pearson_r | spearman_rho | r2 | n |
| --- | --- | --- | --- | --- | --- | --- |
| v4_group_cv_oof | catboost | physchem+esm2 | 0.352 | 0.381 | 0.092 | 2323.000 |
| v4_group_cv_oof | extratrees | physchem+esm2 | 0.335 | 0.378 | 0.082 | 2323.000 |
| v4_group_cv_oof | svr | physchem+esm2 | 0.327 | 0.347 | 0.052 | 2323.000 |
| v4_group_cv_oof | xgboost | physchem+esm2 | 0.316 | 0.351 | 0.068 | 2323.000 |
| v4_group_cv_oof | lightgbm | physchem+esm2 | 0.284 | 0.327 | 0.042 | 2323.000 |
| v4_group_cv_oof | tobit_pca64 | physchem+esm2 | 0.223 | 0.249 | -0.836 | 2323.000 |
| deep_oof_or_val | cnn1d | sequence | -0.079 | -0.079 | -1.276 | 2323.000 |
| deep_oof_or_val | multiscale_cnn | sequence | 0.046 | 0.061 | -1.301 | 2323.000 |
| deep_oof_or_val | bilstm | sequence | 0.210 | 0.207 | -0.951 | 2323.000 |
| deep_oof_or_val | esm2_35M_ft | esm2_t12_35M | 0.398 | 0.418 | -0.182 | 312.000 |

## Aqueous / buffer solubility — cluster CV

| solvent_model | n | n_clusters | cv_roc_auc | cv_pr_auc |
| --- | --- | --- | --- | --- |
| ultrapure_water | 1326 | 1191 | 0.782 | 0.823 |
| 01_m_pbs | 635 | 534 | 0.773 | 0.738 |
| 1x_dpbs | 667 | 639 | 0.779 | 0.623 |
| pooled | 2664 | 1197 | 0.785 | 0.766 |

Endpoint: binary soluble/insoluble under named aqueous/buffer solvent
(SolPepBench / PepSol2000). **Not** E. coli expression.

## Selected production stack

| Component | Choice | Why |
| --- | --- | --- |
| HC50 regressor | ESM-2 35M fine-tune + censored head | Best val pearson (0.398) vs CatBoost OOF (0.352); best locked RC consistency (0.807) |
| HC50 classifier | ExtraTrees physchem+ESM-2 + isotonic | Calibrated P(HC50 > 128 µM) |
| Solubility | SolPepBench pooled aqueous (default 0.1 M PBS) | User-required aqueous/buffer endpoint; pooled CV AUC 0.785 |
| Stability | PEPlife2 protease RF | Best available labelled half-life assay head |
