import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def prepare_data(esg_scores, stock_returns, market_values):
    """准备数据，合并评分和收益率，并使用前一年的ESG评分"""
    # 去除股票代码的后缀
    esg_scores['stkcd'] = esg_scores['stkcd'].str.replace(r'\.\w+', '', regex=True)
    # 调整股票收益率数据中的年份
    stock_returns['year'] = pd.to_datetime(stock_returns['date'], errors='coerce').dt.year

     # 确保'year'列是整数类型
    esg_scores['year'] = esg_scores['year'].astype(int)
    market_values['year'] = market_values['year'].astype(int)

    # 确保'stkcd'列在两个DataFrame中数据类型相同
    esg_scores['stkcd'] = esg_scores['stkcd'].astype(str)
    market_values['stkcd'] = market_values['stkcd'].astype(str)

    # 合并ESG评分和市值
    esg_with_mv = pd.merge(esg_scores, market_values, on=['stkcd', 'year'], how='left')
    # 合并ESG评分和市值数据到股票收益率数据中
    merged_data = pd.merge(stock_returns, esg_with_mv, on=['stkcd', 'year'], how='left')

    return merged_data


def calculate_adjusted_escores(df, year):
    """计算特定年份市值加权的E_Score"""
    # 筛选特定年份的数据
    annual_data = df[df['year'] == year].copy()
    # 计算市值加权的E_Score
    total_market_value = annual_data['marketvalue'].sum()
    annual_data['Adjusted_E_Score'] = (annual_data['e_score'] * annual_data['marketvalue']) / total_market_value
    
    return annual_data


def select_stocks_for_portfolio(df):
    """选择每年E_Score排名前后10%的股票"""
    df['E_Score_Rank'] = df['Adjusted_E_Score'].rank(pct=True)
    long_stocks = df[df['E_Score_Rank'] > 0.8]
    short_stocks = df[df['E_Score_Rank'] < 0.2]
    return long_stocks, short_stocks


def calculate_portfolio_returns_series(df):
    """计算并返回多空组合的每期收益序列"""

    # 处理多头部分
    if 'long' in df.columns and 'chg' in df.columns:
        long = df.loc[df['long'] == 1, ['date','name','chg']].reset_index()
        long = long[~long['name'].str.contains('ST\*?')]
        long_returns = long[['date','chg']].groupby('date').mean()
        
    # 处理空头部分
    if 'short' in df.columns and 'chg' in df.columns:
        short = df.loc[df['short'] == 1, ['date','name','chg']].reset_index()
        short = short[~short['name'].str.contains('ST\*?')]
        short_returns = short[['date','chg']].groupby('date').mean()    

    ret = pd.merge(long_returns, short_returns, on='date', suffixes=('_long', '_short'))

    return ret


def run_longshort_strategy(esg_scores, stock_returns, mv, output_path='中间输出/组合收益/e_score收益.xlsx'):
    """执行长短策略并绘制收益图"""
    data = prepare_data(esg_scores, stock_returns, mv)
    
    ret_=[]

    years = pd.Series(data['year'].unique()).dropna().tolist()
    # years = [2018,2019,2020,2021,2022]
    for year in tqdm(years, desc="处理进度"):
        adjusted_scores = calculate_adjusted_escores(data, year)
        long_stocks, short_stocks = select_stocks_for_portfolio(adjusted_scores)

        long_stocks = long_stocks.copy()
        short_stocks = short_stocks.copy()

        long_stocks.loc[:, 'long'] = 1
        short_stocks.loc[:, 'short'] = 1
        portfolio_data = pd.concat([long_stocks, short_stocks])

        ret = calculate_portfolio_returns_series(portfolio_data)
        ret_.append(ret)
        

    ret_ = pd.concat(ret_)

    ret_['long_cum_ret'] = (ret_['chg_long'].astype('float')+1).cumprod()
    ret_['short_cum_ret'] = (ret_['chg_short'].astype('float')+1).cumprod()
    ret_.to_excel(output_path)
    plot_performance(ret_)


def plot_performance(ret, output_path='中间输出/组合收益/e_score收益.png'):
    """绘制长短策略的绩效图"""
    plt.figure(figsize=(10, 6))
    ret.index = pd.to_datetime(ret.index, format='%Y-%m-%d')
    plt.plot(ret.index, ret['long_cum_ret'], color='green',  linestyle='-', label='Long Portfolio')
    plt.plot(ret.index, ret['short_cum_ret'], color='red',  linestyle='--', label='Short Portfolio')

    plt.title('GMB策略收益')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.grid(True)
    plt.legend()  # This adds the legend to differentiate long and short portfolios
    plt.savefig(output_path)

