import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm


def simulate_counterfactual_performance(data, dependent_var, independent_vars, date_col, window_size=240, bootstrap_samples=500, output_path=None):
    # 将日期转换为 datetime，确保正确排序和滚动窗口计算
    data[date_col] = pd.to_datetime(data[date_col])
    data.sort_values(date_col, inplace=True)

    # 初始化累计计算结果的 DataFrame
    cumulative_results = pd.DataFrame()
    y = data[dependent_var]

    # 以月度滚动窗口进行迭代
    for start in tqdm(range(0, len(data), window_size),desc='滚动估计中'):
        window_data = data.iloc[start:start + window_size].copy()
        X = window_data[independent_vars] 
        y2 = y.iloc[start:start + window_size]

        X = sm.add_constant(X)  # 添加常数项

        # 建立并拟合OLS回归模型
        model2 = sm.OLS(y2, X).fit()

        # 计算基本反事实表现：残差 + 常数项
        window_data.loc[:, 'Counterfactual'] = model2.params['const'] + model2.resid

        # 抽样系数并计算置信区间
        coef_samples = np.random.multivariate_normal(mean=model2.params-0.0019, cov=model2.cov_params(), size=bootstrap_samples)
        counterfactual_samples = []

        for params in coef_samples:
            simulated_y = np.dot(X, params)
            counterfactual_samples.append(simulated_y)

        # 计算每个点的置信区间
        lower_bounds = np.percentile(counterfactual_samples, 10.0, axis=0)
        upper_bounds = np.percentile(counterfactual_samples, 90.0, axis=0)

        # 更新结果 DataFrame
        results = pd.DataFrame({
            'Date': window_data[date_col],
            'Reality': y2,
            'Cumulative Counterfactual': window_data['Counterfactual'],
            'Cumulative Lower CI': lower_bounds,
            'Cumulative Upper CI': upper_bounds
        })

        cumulative_results = pd.concat([cumulative_results, results], ignore_index=True)

    cumulative_results = (cumulative_results.set_index('Date')+1).cumprod()
    cumulative_results.to_excel('输出/反事实表现/反事实表现.xlsx')
    return cumulative_results


def draw_counterfactual(cumulative_results, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_results.index, cumulative_results['Cumulative Counterfactual'], label='Cumulative Counterfactual', color='blue')
    # plt.plot(cumulative_results.index, cumulative_results['Cumulative Lower CI'], label='Lower CI', linestyle='--', color='green')
    # plt.plot(cumulative_results.index, cumulative_results['Cumulative Upper CI'], label='Upper CI', linestyle='--', color='red')
    plt.plot(cumulative_results.index, cumulative_results['Reality'], label='Realized', color='black')
    plt.legend()
    plt.title('Cumulative Counterfactual Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Performance')
    if output_path:
        plt.savefig(output_path)
    plt.show()