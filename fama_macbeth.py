import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import re


def preprocess_data(returns, factors):
    # 数据清洗与转换
    # 获取重复项的索引
    duplicate_indices = returns[returns.duplicated(subset=['date', 'stkcd'], keep=False)].index
    # 删除这些索引对应的行
    returns.drop(index=duplicate_indices, inplace=True)

    returns = returns.pivot(index='date', columns='stkcd', values='chg').astype(float)
    returns.index = pd.to_datetime(returns.index)

    if 'date' in factors.columns:
        factors.set_index('date', inplace=True)
    factors = factors.astype(float)
    factors = sm.add_constant(factors)  # 提前加常数项
    
    common_dates = pd.to_datetime(returns.index.intersection(factors.index))
    aligned_ret = returns.loc[common_dates]
    aligned_factors = factors.loc[common_dates]

    return len(common_dates), aligned_ret, aligned_factors


def perform_rolling_regression(rets, factors, n_periods, window):
    betas = []  # 用于存储每个窗口的beta系数

    for start in tqdm(range(0, n_periods - window + 1, 1), desc='滚动回归计算beta'):
        end = start + window
        ret_window = rets.iloc[start:end]
        factors_window = factors.iloc[start:end]
        
        # 用于存储当前窗口的beta系数
        temp_betas = []
        
        try:
            X = sm.add_constant(factors_window)  # 添加常数项
            for column in ret_window.columns:
                y = ret_window[column].dropna()
                y, X = y.align(X, join='inner', axis=0)  # 数据对齐

                if not y.empty and not X.empty:
                    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                    temp_betas.append(pd.Series(beta, name=column, index=X.columns))
        except np.linalg.LinAlgError as e:
            print(f"无法拟合 {start}-{end}: {e}")
            # 如果发生错误，使用NaN填充
            temp_betas.append(pd.Series([np.nan] * X.shape[1], index=X.columns, name=column))
        
        # 存储当前窗口的结果
        if temp_betas:
            betas.append(pd.concat(temp_betas, axis=1))

    betas_df = pd.concat(betas, axis=0)
    return betas_df


def perform_section_regresssion(rets, betas, n_periods, window):
    results_data = []
    
    for i, date in enumerate(range(window, n_periods, 1)):
        y_cross = rets.iloc[date].dropna()
        beta_df = betas[betas['period']==i+1].set_index('factor_name').T
        beta_df.drop('const', axis=1, inplace=True)
        X_cross = sm.add_constant(beta_df).dropna()  # 添加常数项
        y_cross, X_cross = y_cross.align(X_cross, join='inner', axis=0) 

        if X_cross.empty or y_cross.empty:
            continue

        model_cross = sm.OLS(y_cross, X_cross)
        results_cross = model_cross.fit()
        
        def get_significance_stars(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''

        for var_name in results_cross.params.index:
            stars = get_significance_stars(results_cross.pvalues[var_name])
            results_data.append({
                'Test': f"{i+1}",
                'Variable': var_name,
                'Value': f"{results_cross.params[var_name]:.4f}{stars} ({results_cross.tvalues[var_name]:.2f})"
            })
            
        # 添加R-squared和Observations作为变量行
        results_data.append({
            'Test': f"{i+1}",
            'Variable': 'R-squared',
            'Value': f"{results_cross.rsquared:.2f}"
        })
        results_data.append({
            'Test': f"{i+1}",
            'Variable': 'Observations',
            'Value': f"{n_periods}"
        })

    results_df = pd.DataFrame(results_data)
    pivot_df = results_df.pivot(index='Variable', columns='Test', values='Value')
    
    return pivot_df


def fama_macbeth_regression(ret, factors, window=500, betas=None, output_path=None):
    n_periods, rets, factors = preprocess_data(ret, factors)

    if betas is None:
        betas = perform_rolling_regression(rets=rets,factors=factors,n_periods=n_periods, window=window)
        betas.to_pickle(output_path + '/beta系数.pkl')
    
    # 重置索引，将数据来源的索引转化为列 'factor_name'
    betas.reset_index(inplace=True, names=['factor_name'])

    # 为每个数据来源分配一个递增的 'period'
    # 这里我们用 cumcount() 来为每个 'factor_name' 分组内的行分配一个递增的编号
    betas['period'] = betas.groupby('factor_name').cumcount() + 1

    results_df = perform_section_regresssion(rets=rets, betas=betas,n_periods=n_periods, window=window)
    results_df.to_excel(output_path + '/回归结果.xlsx')


def analyze_significance(data, cycle_length=360):
    results = {
        '周期': [],
        '因子名称': [],
        '显著性比例': [],
        '平均系数': [],
        '正系数比例': []
    }

    # Ensure the data is sorted by column labels numerically (ignoring 'Variable' which should be the index)
    if not data.index.name == 'Variable':
        data.set_index('Variable', inplace=True)
    sorted_columns = sorted(data.columns, key=lambda x: int(x))
    data = data[sorted_columns]

    num_samples = len(data.columns)
    cycle_number = 0

    # Processing each cycle
    for start in range(0, num_samples, cycle_length):
        end = min(start + cycle_length, num_samples)
        cycle_columns = data.columns[start:end]  # Get the correct columns for the current cycle
        cycle_data = data[cycle_columns]
        cycle_number += 1

        # Process each row (factor) within the current cycle
        for index, row in cycle_data.iterrows():
            if index not in ['Observations', 'R-squared']:
                # Cleaning and converting data
                cleaned_row = row.astype(str).replace(r'\*+', '', regex=True).replace(r'\s*\([^)]*\)', '', regex=True)
                row_numeric = pd.to_numeric(cleaned_row, errors='coerce')

                significant = row.str.contains('\*').sum()
                total = row.count()
                proportion_significant = significant / total if total != 0 else 0
                mean_coefficient = row_numeric.mean()
                positive_significant = (row_numeric > 0).sum()
                proportion_positive = positive_significant / total if total != 0 else 0

                results['周期'].append(cycle_number)
                results['因子名称'].append(index)
                results['显著性比例'].append(proportion_significant)
                results['平均系数'].append(mean_coefficient)
                results['正系数比例'].append(proportion_positive)

    return pd.DataFrame(results)

