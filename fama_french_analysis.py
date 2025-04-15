import pandas as pd
import numpy as np
import statsmodels.api as sm

def adjust_scales(df, target):
    # 确定目标列的数量级
    target_scale = np.log10(target.abs().max())
    
    # 调整其他列的数量级以匹配目标列
    for col in df.columns:
        current_scale = np.log10(df[col].abs().max())
        scale_factor = 10**(target_scale - current_scale)
        df.loc[:, col] *= scale_factor.values
    
    return df


def perform_rolling_regression(ret, factors, window=360, output_path='输出/FF回归/回归结果.xlsx'):
    results_data = []
    # factors = adjust_scales(factors, ret)
    # 滚动窗口回归
    n_periods = len(ret)
    for i, start in enumerate(range(0, n_periods - window + 1, window)):
        end = start + window
        # ret_window = (ret.iloc[start:end] + 1).cumprod()
        ret_window = ret.iloc[start:end]
        factors_window = factors.iloc[start:end]

        # 检查数据完整性
        if ret_window.dropna().empty or factors_window.dropna().empty:
            continue

        # 数据对齐和处理
        ret_window = ret_window.dropna()
        factors_window = factors_window.dropna()

        X = sm.add_constant(factors_window)
        model = sm.OLS(ret_window, X)
        results = model.fit()

        # 收集所有变量的统计信息
        for var_name in results.params.index:
            stars = '*' * (results.pvalues[var_name] < 0.05) + '*' * (results.pvalues[var_name] < 0.01) + '*' * (results.pvalues[var_name] < 0.001)
            results_data.append({
                'Test': f"{i+1}",
                'Variable': var_name,
                'Value': f"{results.params[var_name]:.4f}{stars} ({results.tvalues[var_name]:.2f})"
            })
        
        # 添加R-squared和Observations作为变量行
        results_data.append({
            'Test': f"{i+1}",
            'Variable': 'R-squared',
            'Value': f"{results.rsquared:.4f}"
        })
        results_data.append({
            'Test': f"{i+1}",
            'Variable': 'Observations',
            'Value': f"{len(ret_window)}"
        })

    # 创建DataFrame并重塑为需要的格式
    results_df = pd.DataFrame(results_data)
    pivot_df = results_df.pivot(index='Variable', columns='Test', values='Value')

    # 保存到CSV文件
    pivot_df.to_excel(output_path)

# 示例调用
# perform_rolling_regression(ret, factors)
