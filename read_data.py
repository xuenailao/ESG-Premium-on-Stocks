import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        self._e_score = None
        self._stock_list = None
        self._stock_ret = None
        self._market_value = None
        self._index_close = None
        self._news = None
        self._factors = None
        self._industry = None
        self._ccpu = None

        self.esg_files = ['数据\\股票\\' + file_name for file_name in 
                          ['华证ESG评级.xlsx', 'WindESG评级.xlsx']]
        self.stk_ret_files = '数据\\股票\\股票回报率.pkl'
        # self.stk_ret_files1 = ['数据\\股票\\涨跌幅' + file_name for file_name in 
        #                   ['09', '14', '19']]
        self.stk_markevalue_files = '数据\\股票\\总市值.xlsx'
        self.index_file = '数据\\股票\\指数.xlsx'
        self.news_files = ['数据/新闻/News_NewsInfo.xlsx'] + \
                          [f"数据/新闻/News_NewsInfo({i}).xlsx" for i in range(1, 14)]
        self.factors_files = ['数据/多因子/五因子（UMD）/五因子（UMD）.xlsx',
                              '数据/多因子/fama-french三因子/三因子.csv',
                              '数据/多因子/fama-french五因子/五因子.csv']
        self.industry_file = '数据/股票/行业分类.xlsx'
        self.ccpu_file = '数据/CCPU.xlsx'

    @property
    def e_score(self):
        if self._e_score is None:
            self._e_score = pd.read_excel(self.esg_files[1], sheet_name='score')[['stkcd', 'name', 'year', 'e_score']]
        return self._e_score
    
    @property
    def industry(self):
        if self._industry is None:
            self._industry = pd.read_excel(self.industry_file)
        return self._industry
    
    @property
    def ccpu(self):
        if self._ccpu is None:
            self._ccpu = pd.read_excel(self.ccpu_file)
        return self._ccpu
    
    @property
    def stock_list(self):
        if self._stock_list is None:
            self._stock_list = pd.read_excel(self.esg_files[2], sheet_name='stocks')
            self._stock_list['stkcd'] = self._stock_list['stkcd'].str.replace(r'\.\w\w', '', regex=True)
        return self._stock_list
    
    # @property
    # def stock_ret(self):
    #     if self._stock_ret is None:
    #         # 初始化DataFrame列表
    #         dfs = []

    #         # 遍历每个文件夹
    #         for folder in self.stk_ret_files1:
    #             # 获取文件夹中所有文件和子目录的名称
    #             files = os.listdir(folder)
    #             # 遍历文件名
    #             for file in files:
    #                 # 构建文件的完整路径
    #                 file_path = os.path.join(folder, file)
    #                 # 检查文件是否为Excel文件（假设以'.xlsx'结尾）
    #                 if file_path.endswith('.xlsx'):
    #                     # 读取Excel文件
    #                     df = pd.read_excel(file_path)
    #                     # 将DataFrame添加到列表中
    #                     dfs.append(df)

    #         # 使用pd.concat一次性合并所有DataFrame
    #     if dfs:
    #         df_combined = pd.concat(dfs, ignore_index=True)
    #         # 将合并后的DataFrame输出为pickle文件
    #         df_combined.to_pickle("股票回报率.pkl")

    #     return self._stock_ret
    
    @property
    def stock_ret(self):
        if self._stock_ret is None:
            self._stock_ret = pd.read_pickle(self.stk_ret_files)
            self._stock_ret = self._stock_ret.drop(self._stock_ret.index[[0, 1]])
            self._stock_ret.columns = ['stkcd','date','chg']
        return self._stock_ret
    
    @property
    def market_value(self):
        if self._market_value is None:
            self._market_value = pd.read_excel(self.stk_markevalue_files, dtype={'stkcd': str})
        return self._market_value

    @property
    def air_quality(self):
        if self._air_quality is None:
            self._air_quality = pd.read_excel(self.air_q_file)
            self._air_quality['date'] = pd.to_datetime(self._air_quality['date'], format='%Y-%m-%d')  # 根据具体格式调整
            self._air_quality['quality'] = self._air_quality['AQI'].combine_first(self._air_quality['API'])
            self._air_quality = self._air_quality[['date','quality']]
        return self._air_quality

    @property
    def index_close(self):
        if self._index_close is None:
            self._index_close = pd.read_excel(self.index_file)
        return self._index_close

    @property
    def news(self):
        if self._news is None:
            news_dfs = [pd.read_excel(file, skiprows=[0, 2]) for file in self.news_files]
            self._news = pd.concat(news_dfs, axis=0)
        return self._news
              
    def factors(self,n):
        if n == 'ff3':
            return pd.read_csv(self.factors_files[1])
        elif n == 'ff5':
            factors = pd.read_csv(self.factors_files[2])
            factors = factors[factors['Portfolios']==1].drop('Portfolios', axis=1)
            
            return factors
        
        elif n == 'umd5':
            return pd.read_excel(self.factors_files[0])


class DataLoader2:
    def __init__(self):
        self._textfactors = None
        self._escore_return = None
        self._betas = None
        self._weights = None
        self._fm_betas = None
        self._fm_results = None

        self.textfactors_file = '输出\新闻处理\主题关注度.xlsx'
        self.escore_return_file = '输出\E评分组合收益\e_score收益.xlsx'
        self.fm_ff3_file = '输出/FM回归/ff3/'
        self.fm_ff5_file = '输出/FM回归/ff5/'
        self.fm_umd5_file = '输出/FM回归/umd5/'
        self.weights_file = '输出/组合beta/组合权重.xlsx'


    @property
    def textfactors(self):
        if self._textfactors is None:
            self._textfactors = pd.read_excel(self.textfactors_file, sheet_name='关注度')
            self._textfactors = self._textfactors[['公布日期','Topic_0','Topic_1','Topic_9']]
            self._textfactors.columns = ['date','碳排放','环境监管','绿色金融']
        return self._textfactors
    
    @property
    def escore_return(self):
        if self._escore_return is None:
            self._escore_return = pd.read_excel(self.escore_return_file)
        return self._escore_return

    def betas(self, n):
        betas_file = f'输出/组合beta/beta系数_{n}天.xlsx'
        self._betas = pd.read_excel(betas_file, dtype={1: str})
        
        return self._betas

    def fm_results(self, type, type2):
        if type == 'ff3':
            fm_results_file = self.fm_ff3_file
        if type == 'ff5':
            fm_results_file = self.fm_ff5_file
        if type == 'umd5':
            fm_results_file = self.fm_umd5_file

        if type2=='beta':
            self._fm_betas = pd.read_pickle(fm_results_file+'beta系数.pkl')
            return self._fm_betas
        if type2=='result':
            self._fm_results = pd.read_excel(fm_results_file+'回归结果.xlsx')
            return self._fm_results
        
    @property
    def weights(self):
        self._weights = pd.read_excel(self.weights_file)
        return self._weights
