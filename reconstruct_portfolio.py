import pandas as pd
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm


def compute_rolling_betas(factors, ret, window=120, output_path='输出/组合beta'):
    # 转换日期格式并对齐索引
    factors.index = pd.to_datetime(factors.index)
    ret.index = pd.to_datetime(ret.index, errors='coerce')

    # 初始化结果存储字典
    betas = {stock: [] for stock in ret['stkcd'].unique()}
    
    # 遍历每个股票进行回归分析
    for stock in tqdm(ret['stkcd'].unique(), desc='Computing betas for each stock'):
        # 获取特定股票的收益率数据
        ret_stk = ret[ret['stkcd'] == stock]['chg'].copy()
        n_periods = len(ret_stk)

        # 滚动窗口
        for start in range(0, n_periods - window + 1, window):
            end = min(start + window, n_periods)

            # 计算当前窗口的结束日期
            end_date = ret_stk.index[end - 1]  # 防止越界，end-1
            
            # 计算累计收益
            current_ret = (ret_stk.iloc[start:end] + 1).cumprod()

            # 获取对应时间段的因子数据
            current_factors = factors.reindex(current_ret.index)

            # 清理数据
            if not current_factors.empty:
                current_factors = current_factors.apply(pd.to_numeric, errors='coerce').dropna()
                current_ret = pd.to_numeric(current_ret, errors='coerce').dropna()

                # 确保因子和收益率数据索引完全对齐
                if not current_factors.index.equals(current_ret.index):
                    continue

                # 添加常数项以拟合截距
                X = sm.add_constant(current_factors)
                y = current_ret.reindex(X.index)

                # 进行回归分析
                if len(y) > 10:
                    try:
                        model = sm.OLS(y, X, missing='drop').fit()
                        betas[stock].append((model.params.drop('const'), end_date))
                    except Exception as e:
                        print(f"Failed to fit model for {stock} from {start} to {end}: {e}")
                else:
                    betas[stock].append((pd.Series([np.nan]*len(factors.columns), index=factors.columns), end_date))
            else:
                betas[stock].append((pd.Series([np.nan]*len(factors.columns), index=factors.columns), end_date))
    
    # 创建DataFrame以存储所有beta值
    beta_records = []
    for stock, data in betas.items():
        for params, date in data:
            beta_records.append(params.rename((date, stock)))

    # 转换记录为DataFrame
    beta_df = pd.DataFrame(beta_records)
    
    # 保存至Excel文件
    path = output_path+f'/beta系数_{window}天.xlsx'
    beta_df.to_excel(path)
    
    return beta_df


def calculate_portfolio_returns(ret, beta_df, freq=120, stock_list=None, output_path='输出/组合beta'):
    # 数据准备和索引设置
    if 'date' in beta_df.columns:
        beta_df['date'] = pd.to_datetime(beta_df['date'])
        beta_df.set_index(['date', 'stkcd'], inplace=True)
        
    if 'date' in ret.columns:
        ret['date'] = pd.to_datetime(ret['date'], errors='coerce')
        ret.set_index(['date', 'stkcd'], inplace=True)
    
    beta_df.sort_index(level=[0, 1], inplace=True)
    ret.sort_index(level=[0, 1], inplace=True)

    if stock_list is not None:
        beta_df = beta_df.loc[beta_df.index.get_level_values('stkcd').isin(stock_list)]

    if stock_list is None or not isinstance(stock_list, (list, tuple, set)):
        stock_list = beta_df.index.get_level_values('stkcd').astype(str).unique().tolist()
    
    ret = ret.loc[ret.index.get_level_values('stkcd').isin(stock_list)]

    # 计算所有因子的权重
    weights_df = calculate_all_weights(beta_df)
    # 对 weights_df 进行重新索引并进行后向填充
    weights_df_fit = weights_df.reindex(ret.index, fill_value=np.nan)
    weights_df_fit = weights_df_fit.groupby(level='stkcd').ffill()
    # 将 NaN 值填充为0
    weights_df_fit.fillna(0, inplace=True)

    # if freq == 120:
    #     weight_out_put = weights_df_fit.groupby(level='stkcd').mean()
    #     weight_out_put.to_excel(output_path+'/组合权重.xlsx')


    # 向量化计算返回
    portfolio_returns = pd.DataFrame(index=beta_df.index.get_level_values('date').unique(), columns=beta_df.columns)
    for factor in tqdm(beta_df.columns,desc='因子处理中'):
        # 获取所有权重
        weights = weights_df_fit[factor]
        # 使用矩阵运算来替代循环
        portfolio_returns[factor] = (ret['chg'].mul(weights)).groupby(level='date').sum()
        
    if output_path is not None:
        path = output_path+f'/组合收益_{freq}天.xlsx'
        portfolio_returns.to_excel(path)
    return portfolio_returns


def calculate_all_weights(beta_df):
    # 创建一个空的 DataFrame 来存储权重，索引和列与 beta_df 相同
    weights_df = pd.DataFrame(index=beta_df.index, columns=beta_df.columns)

    # 遍历所有日期和因子
    for date in tqdm(beta_df.index.get_level_values('date').unique(), desc='权重计算中'):
        for factor in beta_df.columns:
            # 提取当前日期和因子的所有 beta 值
            current_betas = beta_df.loc[date, factor]

            # 计算 beta_prime（beta 减去其平均值）
            beta_prime = current_betas - current_betas.mean()
            
            # 计算权重
            weights = beta_prime / len(beta_prime) /np.abs(beta_prime).sum()
            weights = weights.values

            # 存储计算出的权重到 weights_df DataFrame 中
            weights_df.loc[(date, slice(None)), factor] = weights

    return weights_df


def filter_stocks_by_market_value(market_value, year, percentile_low=30, percentile_high=30):
    """
    根据市值大小筛选出指定年份前百分之 percentile_low 和后百分之 percentile_high 的股票。
    :param market_value: DataFrame，包含股票的市值数据，包括年份。
    :param year: int，指定要筛选的年份。
    :param percentile_low: float，低端市值股票的百分比。
    :param percentile_high: float，高端市值股票的百分比。
    :return: tuple，包含两个DataFrame，分别为高市值和低市值的股票。
    """
    # 筛选指定年份的市值数据
    annual_market_value = market_value[market_value['year'] == year]
    
    # 计算百分位数阈值
    low_threshold = np.percentile(annual_market_value['market_value'], percentile_low)
    high_threshold = np.percentile(annual_market_value['market_value'], 100 - percentile_high)
    
    # 筛选股票
    low_value_stocks = annual_market_value[annual_market_value['market_value'] <= low_threshold]
    high_value_stocks = annual_market_value[annual_market_value['market_value'] >= high_threshold]
    
    l_stk_list = low_value_stocks['stkcd'].unique().tolist()
    h_stk_list = high_value_stocks['stkcd'].unique().tolist()

    return l_stk_list, h_stk_list


def calculate_portfolio_returns_by_year(ret, beta_df, market_value, freq=120, output_path='输出/组合beta'):
    """
    根据特定年份计算组合收益。
    :param ret: DataFrame，包含股票的收益率数据。
    :param beta_df: DataFrame，包含股票的beta系数。
    :param year: int，指定的年份列表
    :param freq: int, 回归频率。
    :param output_path: str, 结果输出路径。
    :return: DataFrame，组合收益。
    """
    years = range(2009,2023)

    portl = []
    porth = []
    # 数据准备和索引设置
    if 'date' in beta_df.columns:
        beta_df['date'] = pd.to_datetime(beta_df['date'])  
    else:
        beta_df.reset_index(inplace=True)

    if 'date' in ret.columns:
        ret['date'] = pd.to_datetime(ret['date'], errors='coerce')  
    else:
        ret.reset_index(inplace=True)

    for year in years:
        l_stk, h_stk = filter_stocks_by_market_value(market_value=market_value, year=year)
        
        beta_df_year = beta_df[beta_df['date'].dt.year == year]
        ret_year = ret[ret['date'].dt.year == year]

        port_returns_l =  calculate_portfolio_returns(ret_year, beta_df_year, stock_list=l_stk, freq=freq, output_path=None)
        port_returns_h =  calculate_portfolio_returns(ret_year, beta_df_year, stock_list=h_stk, freq=freq, output_path=None)

        portl.append(port_returns_l)
        porth.append(port_returns_h)

    # 合并所有年份结果
    low_portfolio_returns = pd.concat(portl)
    high_portfolio_returns = pd.concat(porth)

    path = f"{output_path}/低市值重构组合收益_{freq}.xlsx"
    low_portfolio_returns.to_excel(path)

    path = f"{output_path}/高市值重构组合收益_{freq}.xlsx"
    high_portfolio_returns.to_excel(path)

    return low_portfolio_returns, high_portfolio_returns


def industry_analysis(weights, industry, output_path='输出/组合beta/行业分布.xlsx'):
    # 通过股票代码合并数据集
    weights.rename(columns={'碳排放': 'weight'}, inplace=True)
    combined_data = pd.merge(weights, industry, on='stkcd', how='inner')
    
    # 对“碳排放”列进行排序，并选择数据的上下30%
    top_30 = combined_data.nlargest(int(len(combined_data) * 0.3), '碳排放')
    bottom_30 = combined_data.nsmallest(int(len(combined_data) * 0.3), '碳排放')
    
    # 计算行业出现次数和权重和
    top_industry_counts = top_30.groupby('industryname').agg(
        Top_Industry_Count=pd.NamedAgg(column='industryname', aggfunc='count'),
        Top_Weight_Sum=pd.NamedAgg(column='weight', aggfunc='sum')
    ).reset_index()
    bottom_industry_counts = bottom_30.groupby('industryname').agg(
        Bottom_Industry_Count=pd.NamedAgg(column='industryname', aggfunc='count'),
        Bottom_Weight_Sum=pd.NamedAgg(column='weight', aggfunc='sum')
    ).reset_index()
    
    # 合并上下两个数据集
    result = pd.merge(top_industry_counts, bottom_industry_counts, on='industryname', how='outer')
    
    # 如果某个行业在上或下不存在，则填充0
    result.fillna(0, inplace=True)
    
    # 将结果保存为Excel
    result.to_excel(output_path, index=False)
    return result