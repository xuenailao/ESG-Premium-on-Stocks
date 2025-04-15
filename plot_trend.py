import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
font_path = 'C:/Windows/Fonts/simhei.ttf' 
font_prop = FontProperties(fname=font_path)

# 更新 matplotlib 全局字体设置
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False 


def plot_trend_func(data1, data2, rows=None, cols=None,output_path='结果/变量走势.png'):
    """
    绘制两个DataFrame的列之间的关系图。

    """
    if 'date' in data1.columns:
            data1.set_index('date', inplace=True)
    if 'date' in data2.columns:
        data2.set_index('date', inplace=True)

    if not isinstance(data1.index, pd.DatetimeIndex):
            data1.index = pd.to_datetime(data1.index)

    # 检查并转换data2索引
    if not isinstance(data2.index, pd.DatetimeIndex):
        data2.index = pd.to_datetime(data2.index)
    
    n_cols_data1 = data1.shape[1]  # 数据1的列数
    n_cols_data2 = data2.shape[1]  # 数据2的列数
        
    if rows == None and cols == None:      
        # 计算合适的子图布局
        rows = max(n_cols_data1, n_cols_data2)
        cols = min(n_cols_data1, n_cols_data2)

     # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 *rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 遍历所有列，绘制图形
    for i, col1 in enumerate(data1.columns):
        for j, col2 in enumerate(data2.columns):
            ax = axes[i * n_cols_data2 + j]
            
            # 绘制data1的数据
            color = 'tab:blue'
            ax.plot(data1.index, data1[col1], label=f"{col1}", color=color)
            ax.set_ylabel(col1, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            
            # 创建一个共享同一x轴的第二y轴
            ax2 = ax.twinx()
            
            # 绘制data2的数据
            color = 'tab:red'
            ax2.plot(data2.index, data2[col2], label=f"{col2}", linestyle='--', color=color)
            ax2.set_ylabel(col2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 设置图例和标题
            ax.set_title(f"{col1} vs {col2}")
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
    plt.tight_layout()  # 增加子图间的padding
    myfig = plt.gcf() 
    myfig.savefig(output_path)
    plt.close()


