import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from read_data import DataLoader
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

import jieba
from gensim.models import LdaMulticore, LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud

# 绘制相关性矩阵图
import networkx as nx


out_path = '中间输出/新闻处理/'

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().strip().split())
    return stopwords


def preprocess_text(df):
    df['text'] = df['新闻标题'] + " " + df['正文']
    stop_words = load_stopwords("数据/停用词.txt")
    remove_words = {'责任编辑', '记者', '编辑', '新闻', '报道', '点击', '页面', '版权', '新华社',
                    '投资者', '项目', '行业', '企业', '市场', '经济', '公司', '股东', '股份'}

    def fenci_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        words = jieba.cut(text)
        return ' '.join(word for word in words if word not in stop_words and word not in remove_words and not word.isdigit())

    df['processed_text'] = df['text'].apply(fenci_text)
    return df


def get_best_LDA(df, max_topics=30, output_path=out_path+'/lda/'):
    df = preprocess_text(df)
    texts = [text.split() for text in df['processed_text']]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_scores = []
    n_topics_list = range(15, max_topics + 1)
    

    for n_topics in tqdm(n_topics_list, desc="Training LDA Models"):
        lda = LdaMulticore(corpus=corpus, num_topics=n_topics, id2word=dictionary, passes=15)
        coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)

    draw_coherence(n_topics_list, coherence_scores)

    best_n_topics = n_topics_list[coherence_scores.index(max(coherence_scores))]
    best_lda = LdaModel(corpus=corpus, num_topics=best_n_topics, id2word=dictionary, passes=15, iterations=400, random_state=0)
    best_lda.save(os.path.join(output_path, "best_lda.model"))
    dictionary.save(os.path.join(output_path, "best_lda.dict"))

    return best_lda, dictionary


def draw_coherence(n_topics_list, coherence_scores, output_path = out_path+'主题一致性得分.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(n_topics_list, coherence_scores, marker='o')
    plt.xlabel('主题数')
    plt.ylabel('一致性得分')
    plt.title('LDA主题筛选')
    plt.xticks(n_topics_list)
    plt.grid(True)
    plt.savefig(output_path)
    

def draw_wordcloud(lda, dictionary, topic_index, base_path=out_path+'词云'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 获取特定主题下的词及其概率
    topic_terms = lda.get_topic_terms(topic_index, topn=100)
    
    # 使用 gensim 的 Dictionary 来获取词汇的字符串形式
    word_probs = {dictionary[id]: prob for id, prob in topic_terms}
    
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100, font_path='simhei.ttf')
    wc.generate_from_frequencies(word_probs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    save_path = os.path.join(base_path, f'topic_{topic_index}_wordcloud.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Word cloud saved to {save_path}")


def get_attention_index(df, lda_path=out_path+'/lda', output_path=out_path+'主题关注度.xlsx'):
    if not os.path.exists(lda_path):
        raise FileNotFoundError("LDA model file not found.")
    
    # 预处理文本
    df = preprocess_text(df)

    # 加载LDA模型
    lda = LdaModel.load(lda_path+'/best_lda.model')
    dictionary = Dictionary.load(lda_path+'/best_lda.dict')
    
    # 转换文本为词袋表示
    texts = df['processed_text'].apply(lambda x: x.split())
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 获取每个文档的主题概率分布
    topic_probs = [lda.get_document_topics(bow, minimum_probability=0.0) for bow in corpus]
    # 将主题概率列表转换为稠密的矩阵形式
    dense_topic_probs = np.array([[prob for _, prob in doc] for doc in topic_probs])
    topic_df = pd.DataFrame(dense_topic_probs, columns=[f'Topic_{i}' for i in range(lda.num_topics)])

    # 将日期转换为日期时间格式并合并数据
    df['公布日期'] = pd.to_datetime(df['公布日期'])
    df.reset_index(drop=True, inplace=True)
    topic_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, topic_df], axis=1)

    # 聚合每天的主题关注度
    daily_topics = df.groupby('公布日期')[[f'Topic_{i}' for i in range(lda.num_topics)]].sum().reset_index()

    # 计算相关系数矩阵
    corr_matrix = daily_topics.drop('公布日期', axis=1).corr()
    up_corr = pd.DataFrame(np.triu(corr_matrix, k=1), columns=corr_matrix.columns, index=corr_matrix.index)

    # 保存到Excel
    with pd.ExcelWriter(output_path) as writer:
        daily_topics.to_excel(writer, sheet_name='关注度', index=False)
        up_corr.to_excel(writer, sheet_name='相关性')

    return daily_topics, corr_matrix


def plot_topic_correlation_network(corr_matrix, threshold=0.7, layout='spring', title='Topic Correlation Network',
                                    output_path = out_path+'相关系数图.png'):
    """
    绘制主题之间的相关性网络图。
    
    参数:
    - corr_matrix (DataFrame): 主题之间的相关性矩阵。
    - threshold (float): 添加边的相关性阈值。
    - layout (str): 图的布局类型（'spring', 'circular', 'random', 'shell' 等）。
    - title (str): 图的标题。
    """
    G = nx.Graph()
    
    # 添加节点
    for topic in corr_matrix.columns:
        G.add_node(topic)
    
    # 添加边
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] > threshold:
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])
    
    # 选择布局
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)  # 默认使用spring layout

    # 获取边的权重，用于设置边的颜色和宽度
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
    edge_widths = [5 * d['weight'] for _, _, d in G.edges(data=True)]

    # 绘制图形
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors,
            width=edge_widths, node_size=700, font_size=8, font_color='darkred', alpha=0.7)
    
    plt.title(title)
    plt.savefig(output_path)


def main(df):
    #lda_model, dictionary = get_best_LDA(df)
    lda_path = out_path+'/lda'
    lda_model = LdaModel.load(lda_path+'/best_lda.model')
    dictionary = Dictionary.load(lda_path+'/best_lda.dict')
    
    # # 绘制词云
    # for topic_index in range(lda_model.num_topics):  # 使用 range() 来创建可迭代的索引序列
    #     draw_wordcloud(lda=lda_model, dictionary=dictionary, topic_index=topic_index)

    daily_topics, corr_matrix = get_attention_index(df)
    
    plot_topic_correlation_network(corr_matrix, threshold=0.7, layout='spring')

    
if __name__ == '__main__':
    font_path = 'C:/Windows/Fonts/simhei.ttf' 
    font_prop = FontProperties(fname=font_path)

    # 更新 matplotlib 全局字体设置
    plt.rcParams['font.family'] = font_prop.get_name()
    loader = DataLoader()

    news = loader.news
    main(news)

