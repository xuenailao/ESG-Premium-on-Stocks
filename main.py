import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from read_data import DataLoader, DataLoader2
from e_score_factor import run_longshort_strategy
from fama_french_analysis import perform_rolling_regression
from plot_trend import plot_trend_func
from reconstruct_portfolio import compute_rolling_betas, calculate_portfolio_returns, calculate_portfolio_returns_by_year, industry_analysis
from fama_macbeth import fama_macbeth_regression, analyze_significance
from counterfactual import simulate_counterfactual_performance, draw_counterfactual

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
# 设置支持中文的字体路径
font_path = 'C:/Windows/Fonts/simhei.ttf' 
font_prop = FontProperties(fname=font_path)

# 更新 matplotlib 全局字体设置
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False 

os.chdir('D:\Quant_JunruWang\毕业论文代码')
loader = DataLoader() # loader用于读取原始数据
loader2 = DataLoader2() # loader用于读取处理后数据

industry =loader.industry
weights = loader2.weights
industry_analysis(weights, industry)