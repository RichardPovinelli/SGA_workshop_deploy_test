"""Cross-model comparison and analysis functions."""

import pandas as pd
from src.data.split import get_test_df
from src.evaluation.dm_test import diebold_mariano_test
from src.evaluation.utils import round_metric_columns
from src.models.elasticnet import evaluate_elasticnet_model
from src.models.lasso import evaluate_lasso_model
from src.models.linear import evaluate_linear_model
from src.models.naive import evaluate_naive_model
from src.models.xgb import evaluate_xgb


def evaluate_all_baseline_models(df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate all baseline models across all 7 horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset
    test_df : pd.DataFrame
        Test dataset

    Returns
    -------
    pd.DataFrame
        Combined results for naive, linear, lasso, elasticnet models
    """
    all_results = []
    for h in range(1, 8):
        naive_results = evaluate_naive_model(test_df, h)
        _, linear_results = evaluate_linear_model(df, h)
        _, lasso_results = evaluate_lasso_model(df, h)
        _, elasticnet_results = evaluate_elasticnet_model(df, h)

        all_results.extend([naive_results, linear_results, lasso_results, elasticnet_results])

    comparison = pd.concat(all_results, ignore_index=True)
    return round_metric_columns(comparison, ["mape", "wmape", "rmse"], sig=3)


def evaluate_xgboost_vs_linear(df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate XGBoost vs Linear across all 7 horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset
    test_df : pd.DataFrame
        Test dataset

    Returns
    -------
    pd.DataFrame
        Combined results for xgboost and linear models
    """
    all_results = []
    for h in range(1, 8):
        _, xgb_results = evaluate_xgb(df, h)
        _, linear_results = evaluate_linear_model(df, h)

        all_results.extend([xgb_results, linear_results])

    comparison = pd.concat(all_results, ignore_index=True)
    return round_metric_columns(comparison, ["mape", "wmape", "rmse"], sig=3)


def evaluate_all_three_models(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlp_models: dict,
) -> pd.DataFrame:
    """Combine results from all three model families across 7 horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset
    test_df : pd.DataFrame
        Test dataset
    mlp_models : dict
        Pre-trained MLP models keyed by horizon

    Returns
    -------
    pd.DataFrame
        Combined results for xgboost, linear, and mlp_forecast models
    """
    from src.models.mlp_forecast import evaluate_mlp_forecast_model

    # Get XGBoost + Linear comparison (already rounded)
    xgb_linear = evaluate_xgboost_vs_linear(df, test_df)

    # Evaluate MLP models
    mlp_results = []
    for h in range(1, 8):
        mlp_result = evaluate_mlp_forecast_model(test_df, h, model=mlp_models[h])
        mlp_results.append(mlp_result)

    mlp_comparison = pd.concat(mlp_results, ignore_index=True)
    mlp_comparison = round_metric_columns(mlp_comparison, ["mape", "wmape", "rmse"], sig=3)

    # Combine all three
    three_way = pd.concat([xgb_linear, mlp_comparison], ignore_index=True)
    return three_way


def evaluate_all_models_with_tft(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlp_models: dict,
    tft_model: object,
) -> pd.DataFrame:
    """Combine results from all model families including TFT across 7 horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (train + test); required for TFT context window.
    test_df : pd.DataFrame
        Test dataset.
    mlp_models : dict
        Pre-trained MLP models keyed by horizon.
    tft_model : object
        Pre-trained TFT model.

    Returns
    -------
    pd.DataFrame
        Combined results for xgboost, linear, mlp_forecast, and tft models.
    """
    from src.models.tft import evaluate_tft_model

    three_way = evaluate_all_three_models(df, test_df, mlp_models)

    tft_results = []
    for h in range(1, 8):
        tft_result = evaluate_tft_model(df, h, model=tft_model)
        tft_results.append(tft_result)

    tft_comparison = pd.concat(tft_results, ignore_index=True)
    tft_comparison = round_metric_columns(tft_comparison, ["mape", "wmape", "rmse"], sig=3)

    return pd.concat([three_way, tft_comparison], ignore_index=True)


def run_pairwise_dm_tests(
    predictions_dict: dict,
    target: pd.Series,
    model_names: list[str],
) -> pd.DataFrame:
    """Run pairwise one-sided Diebold-Mariano tests between models.

    Parameters
    ----------
    predictions_dict : dict
        Mapping of model names to prediction series
    target : pd.Series
        Actual target values
    model_names : list[str]
        Ordered list of model names to test

    Returns
    -------
    pd.DataFrame
        P-value matrix where (i,j) is p-value for H0: model_i >= model_j
    """
    n_models = len(model_names)
    pvalue_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                pvalue_matrix.loc[m1, m2] = 1.0
            else:
                result = diebold_mariano_test(
                    target,
                    predictions_dict[m1],
                    predictions_dict[m2],
                    alternative="greater",
                )
                pvalue_matrix.loc[m1, m2] = result["p_value"]

    return pvalue_matrix.astype(float)
