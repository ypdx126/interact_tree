# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:01:33 2020

@author: ypdx

交互式决策树、CartTree的实现
"""

import os
import numpy as np
import pandas as pd
# from .core import bad2woe, na_split
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.pyplot import imread, imshow, show
from io import BytesIO


def bad2woe(badrate_i, badrate_base):
    """把 badrate 转换成 woe 值。\n

    参数:
    ---------
    badrate_i: float or 1darray or series, 各个箱的 badrate  \n
    badrate_base: float, 全局的基准 badrate  \n

    返回值:
    ----------
    woe_i: 类型同 badrate_i, 各个箱对应的 woe 值。"""
    return np.log(1 / badrate_i - 1) - np.log(1 / badrate_base - 1)


def na_split(var, y):
    """把目标变量缺失值对应的数据丢弃、特征变量的缺失值对应的数据单独分离出来作为一组 \n

    参数:
    ---------
    var: series, 特征变量 \n
    y: series, 目标变量 \n

    返回值:
    ---------
    na_group: 如果var没有缺失值，np_group返回值为0；否则返回一dataframe  \n
    var_notNa: 分离缺失值后的特征变量  \n
    y_notNa: 分离缺失值后的目标变量   \n"""
    logic = pd.notnull(y)
    var = var[logic]
    y = y[logic]
    logic = pd.isnull(var)
    if logic.any():
        na_grp = pd.DataFrame(columns=[0, 1, 'All', 'badRate', 'binPct'], index=[np.nan])
        var_no = var[~logic]
        y_na, y_no = y[logic], y[~logic]
        na_grp[1] = y_na.sum()
        na_grp['All'] = len(y_na)
        na_grp[0] = na_grp['All'] - na_grp[1]
        na_grp['badRate'] = na_grp[1] / na_grp['All']
        na_grp['binPct'] = na_grp['All'] / len(y)
        return na_grp, var_no, y_no
    return 0, var, y


def distribution(self, dropna=False, sort='index'):
    """计算单个变量的分布, 返回的数据框有两列：cnt(个数)、binPct(占比) \n

    参数:
    ----------
    self: Series 对象。
    dropna: bool, 是否要忽略缺失值
    sort: str, 'index' 表示按变量值的顺序排序，其他任意值表示变量值的个数占比排序"""
    a = self.value_counts(dropna=dropna)
    b = a / a.sum()
    df = pd.DataFrame({'cnt': a, 'binPct': b})
    if sort == 'index':
        df = df.sort_index()
    return df.reindex(columns=['cnt', 'binPct'])


def entropy(sr):
    """计算信息熵，以一个明细的观测点序列为输入  \n

    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比  \n

    返回值:
    ----------
    entr: float, 变量的熵"""
    p = distribution(sr)
    e = p.binPct * np.log(p.binPct)
    return -e.sum()


def gain_entropy(sr, by):
    """计算随机变量的条件熵、gain.  \n

    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比  \n
    by: series, 与 sr 等长，条件明细数据。将按 by 取不同的值分组，计算各组内 sr 的熵，再加权求和  \n

    返回值:
    ----------
    gain_entr: float, 变量的熵增益"""

    entr = entropy(sr)

    d = distribution(by).binPct
    cond_entr = pd.Series(index=d.index)
    for i in d.index:
        ei = entropy(sr[by == i])
        cond_entr[i] = ei
    cond_entr = (cond_entr * d).sum()

    return entr - cond_entr


def gini_impurity(sr):
    """计算基尼不纯度, 以一列明细观测为输入。  \n

    参数:
    ----------
    sr: series, 一列明细数据  \n

    返回值:
    ----------
    impurity: float, 变量的基尼不纯度"""
    p = distribution(sr)
    impurity = 1 - (p.binPct * p.binPct).sum()
    return impurity


def gain_gini_impurity(sr, by):
    """计算条件基尼不纯度、gain。 \n

    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比   \n
    by: series, 与 sr 等长，条件明细数据。将按 by 取不同的值分组，计算各组内 sr 的基尼，再加权求和   \n

    返回值:
    ----------
    gain_gini: float, 变量的基尼增益
    """

    gini = gini_impurity(sr)

    d = distribution(by).binPct
    cond_gini = pd.Series(index=d.index)
    for i in d.index:
        gi = gini_impurity(sr[by == i])
        cond_gini[i] = gi
    cond_gini = (cond_gini * d).sum()

    return gini - cond_gini


def numeric_best_split(sr, y, criteria='gini', min_bins_sample=30):
    """数值型变量、序数型变量的最优二分点

    参数:
    ----------
    sr: series, 数值型或序数型的数据列，
    y: series, 目标变量列
    criteria: str, 按什么指标来计算最优分割，可选值有 'gini', 'entropy'
    min_bins_sample: 二分法时，确保每个子节点的样本容量 >= min_bins_sample

    返回值:
    ----------
    best_split_value: series, 对于给定的类别变量，最优的二分法及分割后的条件熵/条件gini
    """
    def entropy_pi(p_good):
        """以 good 的概率为输入，计算信息熵"""
        if p_good == 0:
            return 0
        else:
            p_bad = 1 - p_good
            return -(p_good * np.log(p_good) + p_bad * np.log(p_bad))

    def gini_pi(p_good):
        """以 good 的概率为输入，计算基尼不纯度"""
        p_bad = 1 - p_good
        return 1 - (p_good**2 + p_bad**2)

    na_group, sr1, y1 = na_split(sr, y)
    if len(sr1) == 0:  # 若sr全部为空
        return pd.Series([0, np.nan, np.nan], index=['gain_' + criteria, 'split_value', 'na_side'])

    gini_base = entropy(y1) if criteria == 'entropy' else gini_impurity(y1)   # 全局基准

    gini_fun = entropy_pi if criteria == 'entropy' else gini_pi
    value_count = pd.crosstab(sr1, y1)
    value_count['all'] = value_count.sum(axis=1)
    if not (0 in value_count.columns):
        value_count[0] = 0
    cum_count = value_count[[0, 'all']].cumsum()
    cum_count.columns = ['cumgood', 'total_left']  # 以 cum_count的index为二分点，提前统计好直方图，方便重复利用该统计快速计算熵、基尼不纯度
    total_good, total = cum_count.iloc[-1]
    cum_count['percent_left'] = cum_count['total_left'] / total   # 左枝的样本量占比
    cum_count['p_good_left'] = cum_count['cumgood'] / cum_count['total_left']   # 二叉树的左枝样本的 p_good
    cum_count['total_right'] = total - cum_count['total_left']   # 右枝样本量
    cum_count['percent_right'] = cum_count['total_right'] / total  # 右枝样本量占比
    cum_count['p_good_right'] = (total_good - cum_count['cumgood']) / cum_count['total_right'] #  右枝样本的 p_good

    gain_max = 0; split_max = np.nan
    for split_value in cum_count.index[:-1]:
        if min(cum_count.at[split_value, 'total_left'], cum_count.at[split_value, 'total_right']) < min_bins_sample:
            continue

        gini_left = gini_fun(cum_count.at[split_value, 'p_good_left'])
        gini_right = gini_fun(cum_count.at[split_value, 'p_good_right'])
        cond_gini = cum_count.at[split_value, 'percent_left'] * gini_left + \
                    cum_count.at[split_value, 'percent_right'] * gini_right
        gain_i = gini_base - cond_gini
        if gain_i > gain_max:
            gain_max = gain_i
            split_max = split_value

    gain_fun = gain_entropy if criteria == 'entropy' else gain_gini_impurity
    best_split_value = pd.Series(index=['gain_'+criteria, 'split_value', 'na_side'], dtype='object')
    best_split_value['split_value'] = split_max
    if len(sr1) != len(sr):
        split_logic = sr < split_max   # 把缺失值放在右边
        gain_gini_right = gain_fun(y, split_logic)

        split_logic = split_logic | pd.isnull(sr)   # 把缺失值放在左边
        gain_gini_left = gain_fun(y, split_logic)

        best_split_value['na_side'] = 'left' if gain_gini_left > gain_gini_right else 'right'
        best_split_value['gain_'+criteria] = max(gain_gini_left, gain_gini_right)
    else:
        best_split_value['gain_'+criteria] = gain_max

    return best_split_value


def category_best_split(sr, y, criteria='gini', min_bins_sample=30):
    """类别型变量分成2个组，使条件 gini 或 entropy 下降最大. 当 y 是二分类问题时，此算法是全局最优的二分类
    当类别型变量的类别个数过多时，可能造成过拟合，应提前把样本数小于阈值（如30）的类别值合并到'other'中去，再传清洗好的数据给本算法
    算法描述：
    1、在本节点下，求类别型变量各组的badrate，并按badrate排序
    2、把排序后的变量当作序数型变量，逐个取值地尝试分成（小于此值，大于等于此值）2组
    3、取分组后条件gini或条件熵最小的分组方案。

    参数:
    ------------
    sr: series, 类别型的数据列，
    y: series, 目标变量列
    criteria: str, 按什么指标来计算最优分割，可选值有 'gini', 'entropy'
    min_bins_sample: 二分法时，确保每个子节点的样本容量 >= min_bins_sample

    返回值:
    -------------
    best_split_value: series, 对于给定的类别变量，最优的二分法及分割后的条件熵/条件gini
    """
    cate = pd.crosstab(sr, y)
    badrate = (cate[1] / cate.sum(axis=1)).sort_values()
    sr1 = sr.as_ordinal(badrate.index)
    best_split_value = numeric_best_split(sr1, y, criteria=criteria, min_bins_sample=min_bins_sample)
    split_max = badrate[:best_split_value['split_value']].index    # 类别型变量的分割值是个列表，列表内是枚举值
    best_split_value['split_value'] = list(split_max)
    return best_split_value

    na_group, sr1, y1 = na_split(sr, y)

    cate = pd.crosstab(sr1, y1)
    cate = cate[1] / cate.sum(axis=1)
    cate = cate.sort_values()
    fun_gain = gain_gini_impurity if criteria == 'gini' else gain_entropy
    
    node_left = []; gini_max = 1000
    for item in cate.index:
        node_left.append(item)
        split_logic = sr1.isin(node_left)
        gini_i = fun_gain(y1, split_logic)
        bins_min = split_logic.value_count().min()
        if gini_i > gini_max and bins_min >= min_bins_sample:  # 逐个加入到左节点，测试条件 gini/entropy 是否会下降
            gini_max = gini_i
        else:
            node_left.pop()

    # 缺失值的处理
    best_split_value = pd.Series(index=['gain_' + criteria, 'split_value', 'na_side'])
    best_split_value['split_value'] = node_left
    if len(sr1) != len(sr):
        split_logic = sr.isin(node_left) | pd.isnull(sr)   # 缺失值放到左节点
        gain_gini_left = fun_gain(y, split_logic)

        split_logic = sr.isin(node_left)    # 缺失值放到右节点
        gain_gini_right = fun_gain(y, split_logic)
        best_split_value['na_side'] = 'left' if gain_gini_left > gain_gini_right else 'right'
        best_split_value['gain_' + criteria] = min(gain_gini_left, gain_gini_right)
    else:
        best_split_value['gain_' + criteria] = gini_max

    return best_split_value


def is_category(serie):
    """是否是类别型的列。
    非数值型的列，都被判定为类别型

    参数:
    ------------
    serie: Series, 数据列

    返回值:
    ------------
    flag: boolean, True 表示是类别型的列"""
    if np.issubdtype(serie.dtype, np.integer) or np.issubdtype(serie.dtype, np.float_):
        return False
    else:
        return True


def best_split(X, y, criteria='gini', min_bins_sample=30):
    """在本层样本数据 node_X, node_y 上，计算各个特征的最优分割点和增益水平,

    参数:
    -----------
    X: dataframe, 预测变量
    y: series or 1darray, 目标变量
    criteria: str, 最优分割点的计算采用哪种指标, 可选有 'gini' or 'entropy'
    min_bins_sample: 最小叶节点的样本数量

    返回值:
    -----------
    feature_split: dataframe, 各个列的最优分割点及分割后的条件熵/条件gini，按升序排列
    """
    feature_split = pd.DataFrame(index=X.columns,
                           columns=['gain_'+criteria, 'split_value', 'na_side'])  # na_side: 计算增益时，nan 值放在哪一边
    for col in X:
        if is_category(X[col]):  # 类别型特征
            split_i = category_best_split(X[col], y, criteria=criteria, min_bins_sample=min_bins_sample)
        else:
            split_i = numeric_best_split(X[col], y, criteria=criteria, min_bins_sample=min_bins_sample)
        feature_split.loc[col] = split_i
    feature_split = feature_split.sort_values('gain_'+criteria, ascending=False)

    return feature_split


class InteractTree:
    def __init__(self, X, y, tree_img_path, criteria='gini', min_leaf_samples=30, node_type='root'):
        """交互式决策树对象，并初始化为根节点。根节点及每个子节点，均是 InteractTree 对象。
        每一次的二分生长，需要交互地、人工设置生长条件。使用 grow 方法来生长子节点。
        每次生长出 left, right 子节点后，均按决策树算法自动计算左右子节点的最优分割特征(split_info_)，并按降序排列；也会自动计算各子节点的统计
        信息(node_info_)。以供人工参考、设定分割的特征和分割点，带入业务知识。
        可以剪枝树的节点，使用 cut_leaf(node_id)，被剪枝节点及其所有子节点都被剪掉。
        使用 root.plot_tree() 来画决策树。

        参数:
        ----------
        X: DataFrame
            训练样本的预测变量. 交给算法的应该是剔除了异常值的、清洗转化好之后的数据，支持类别型、序数型特征，
            支持缺失值，但缺失值能填充的，还是尽量填充好。
            异常的处理：对数值型变量，极大极小值用边界值替代；
            对类别型变量，出现频率小于阈值的统一归类到'other'组中。类别型的列，请将数据类型设置为 pd.CategoricalDtype。
            数值型、序数型的列，设置为 float, int 均可。
        y: Series
            目标变量. 目前只支持二分类的目标变量，取值分别为 0，1。
            y 应与 X 具有相同的 index.
        tree_img_path: str,
            决策树图片的保存路径。绝对路径，不需要带扩展名，会保存成 png 格式
        criteria: str, optional
            决策树的生长条件，可选 gini 或 entropy. 默认使用 gini.
        min_leaf_samples: int,
            决策树的叶节点的最小样本数，若算法选择的最优分割点，将两个子节点中的任一个的样本容量分得小于该阈值，会重新选择分割点
        node_type: 'root' or 'child', 实例化一个根节点还是子节点，除根节点外，其他节点都是子节点。
            只需要手动实例化一个根节点，子节点由对象的 grow 方法自己生成。

        对象属性:
        -----------
        node_id_: int, 节点的唯一编号
        node_info_: series, 节点的统计信息
        split_info_: dataframe, 节点的各变量最优分割点及分割增益，按降序排列。na_side列表示缺失值被分在哪一支，NaN表示该节点无缺失值
        left: 节点的左子节点，None表示该节点是叶节点，无子节点
        right: 节点的右子节点，None表示该节点是叶节点，无子节点
        node_X：dataframe, 节点的子样本数据
        node_y: series, 节点的子目标变量
        level_: int, 该节点处于决策树的第几层，定义根节点是第 0 层

        对象方法:
        ----------
        get_node: 获得给定编号的节点对象
        grow: 生长，即被选中的节点分割一次，分割条件及分割值均由人工给出
        cut_leaf：剪枝，与 grow 相反的操作
        split_rank: 查看所有叶节点的最优分割的排名
        plot_tree: 画出决策树图形。只在根节点上调用此方法才有意义
        generate_tree_code: 生成决策树的 python 代码
        cal_woe_df: 计算决策树对应的 woe_df 表
        is_leaf: 判断节点是否为叶节点
        各方法的详细调用方法，请参考各方法的说明。
        """
        self.__grow_split_str_ = ''  # 用于显示本层分割到下一层的分割逻辑，画树亦用到它. 后续会对此属性赋值
        self.left = None
        self.right = None

        # 初始化根节点
        if node_type == 'root':
            self.X = X
            self.y = y
            self.tree_img_path_ = tree_img_path
            self.criteria_ = criteria
            self.min_leaf_samples_ = min_leaf_samples
            self.base_badrate = y.mean()
            self.base_bad = y.sum()
            self.base_good = len(y) - self.base_bad

            self.__father = None
            self.level_ = 0  # 定义根节点的level = 0
            self.node_id_ = 1
            self.__node_logic = np.ones(len(y)) > 0  # 记录本节点划分的子空间及样本，如何从总训练样本 X 中获取： X[node_logic]
            self.node_info_ = self.__calculate_node_info()
            self.split_info_ = best_split(self.node_X, self.node_y, criteria=criteria, min_bins_sample=min_leaf_samples)

    def __calculate_node_info(self):
        """计算给定节点的相关统计信息, 返回 node_info"""
        node_x = self.node_X
        node_y = self.node_y
        node_info = pd.Series(dtype='object')
        node_info['node_id'] = self.node_id_
        node_info['All'] = len(node_x)
        node_info['badRate'] = node_y.mean()
        root = self.__get_root()
        node_info['binPct'] = node_info['All'] / len(root.X)
        node_info['woe'] = bad2woe(node_info['badRate'], root.base_badrate)
        p_good = (node_info['All'] * (1 - node_info['badRate'])) / root.base_good
        p_bad = node_y.sum() / root.base_bad
        node_info['IV_i'] = (p_good - p_bad) * node_info['woe']

        return node_info

    def __born_child(self, level, node_logic, split_code, node_id):
        """生成并返回一个子节点对象。不能生成根节点。

        参数：
        ----------
        level: int,
            本节点处在决策树的第几层. 根节点是第 0 层。
        node_logic: sr
            本层节点的分割条件, 是与训练数据 X 行维度相等的布尔序列。
        split_code: str
            本层节点的分割代码。供 generate_tree_code方法使用，把每一层的分割代码合并，生成整棵树的代码
        node_id: int
            本层节点的编号。一颗树中，无论是叶节点、中层节点还是根节点，均有唯一的编号

        返回：
        -------
        InteractTree 对象，此对象已经计算好了 node_info_、split_info_"""

        # 其他节点的初始化，与根节点的初始化不一样，需要一个单独的函数
        treenode = InteractTree(None, None, None, node_type='child')
        treenode.__node_logic = node_logic  # 记录本节点划分的子空间及样本，如何从总训练样本 X 中获取： X[node_logic]
        treenode.__split_tree_code = split_code  # 从上一节点分裂到本节点的python分裂代码, 用于自动生成决策树的python代码
        # treenode.__grow_split_str_ = ''  # 用于显示本层分割到下一层的分割逻辑，画树亦用到它
        treenode.level_ = level   # 本节点所处的决策树层级
        treenode.node_id_ = node_id
        treenode.__father = self
        treenode.node_info_ = treenode._InteractTree__calculate_node_info() # 计算node_info需要用到self.node_id, 除根节点外的 node_id都是传进来的
        treenode.split_info_ = best_split(treenode.node_X, treenode.node_y,
                                          criteria=self.__get_root().criteria_,
                                          min_bins_sample=self.__get_root().min_leaf_samples_)  # 计算split_info需要用到__node_logic
        return treenode

    @property
    def node_X(self):
        """返回本节点对应的样本数据"""
        root = self.__get_root()
        return root.X[self.__node_logic]

    @property
    def node_y(self):
        """返回本节点对应的目标变量"""
        root = self.__get_root()
        return root.y[self.__node_logic]

    def grow(self, split_code_string):
        """按指定的分裂条件，生长一层。
        生长条件由 col_name operator split_value 三部分组成，中间以空格隔开。如
        'age <= 28', 'marriage in ["未婚", "离异"]', 'amt is nan' 等均为有效的分裂条件
        每次生长完成，都会自动更新决策树图形到 img_path 所指定的文件名

        参数:
        ----------
        split_code_string: string,
            描述生长条件的文本代码, 只需指定左边子树的分裂条件。
            生长条件由 col_name operator split_value 三部分组成，中间以空格隔开。其中：
            col_name 是分裂时选用的列名，
            operator 是分裂的比较操作符，其取值及解释如下：
                '<', '<=', '!=', '==' 仅用于数字型、序数型的列的比较.因为寻找最优分割时只用 <= 做二分点，因此比较操作符只支持这三个;
                'in' 用于多个值的集合检测，一般是字符型的列的比较；
                'is' 仅用于是否为空的比较，'nan', 'np.nan', 'none', 'null', 'nat' 均是合法的空值;
            split_value 是选择的分裂值。

            若分裂条件是2个条件的组合，用 'and' 或 'or' 连接多个条件, 无须用括号括起各个子条件。如 'age < 28 or age is nan';
            不接受 3 个及以上条件的组合;
            不可以用连续判断（如 a1 <= age < a2， 请改成这种形式： x >= a1 and x < a2 )
            如 'age < 28', 'marriage in ["未婚","离异"]', 'amt is nan' 等均为有效的分裂条件。

        返回值:
        ----------
        无返回值，会生长出 self.left, self.right 两个子节点，并自动画出最新的决策树图形"""
        def analysis_code_string():
            """分析 split_code_string, 并转换成可执行的 python 代码"""
            def single_split_judge(single_code_string):
                """单个完整的分列条件的转化"""
                code_tmp = single_code_string.split(' ')
                col_name = code_tmp[0].strip()  # 分裂的列名
                operator = code_tmp[1].strip()  # 分裂的操作符
                split_value = code_tmp[2].strip()  # 分裂的值

                assert operator in ['<', '<=', '!=', '==', 'in', 'is', '>', '>='], """非法条件判断符号，只接受 '<', '<=', '!=', '==', 'in', 'is' '>', '>='"""
                if operator.find('<') > -1 or operator.find('>') > -1 or operator.find('=') > -1:
                    col_name_eval = 'self._InteractTree__get_root().X["{colname}"] '.format(colname=col_name)
                    code_eval = single_code_string.replace(col_name, col_name_eval)
                elif operator == 'in':
                    code_eval = 'self._InteractTree__get_root().X["{colname}"].isin({value}) '.format(colname=col_name, value=split_value)
                elif operator == 'is':
                    assert split_value.lower() in ('nan', 'np.nan', 'none', 'null', 'nat'), "is 仅用于是否为空的判断，合法的空值有'nan', 'np.nan', 'none', 'null', 'nat'"
                    code_eval = 'pd.isnull(self._InteractTree__get_root().X["{colname}"])'.format(colname=col_name)

                return code_eval

            if split_code_string.find(' or ') > -1:  # 多个条件的组合: or
                codes = split_code_string.split(' or ')
                code_eval = []
                for code_i in codes:
                    eval_i = '({})'.format(single_split_judge(code_i))
                    code_eval.append(eval_i)
                code_eval = ' | '.join(code_eval)

            elif split_code_string.find(' and ') > -1:  # 多个条件的组合: and
                codes = split_code_string.split(' and ')
                code_eval = []
                for code_i in codes:
                    eval_i = '({})'.format(single_split_judge(code_i))
                    code_eval.append(eval_i)
                code_eval = ' & '.join(code_eval)

            else:  # 单个条件
                code_eval = single_split_judge(split_code_string)

            return code_eval

        def cal_next_level_id():
            """计算下一层的两个 node 的 node_id"""
            left_id = self.node_id_ * 2   # 以二进制编码后, 每增加一层，相当于进一位。这种编码不会造成不同节点的node_id重复。
            right_id = left_id + 1
            return left_id, right_id

        def to_tree_code():
            """把 split_code_string 转化为决策树代码，generate_tree_code 在此基础上生成代码"""
            def single_code_trans(single_code):
                """单个分裂条件的决策树代码转换"""
                code_tmp = single_code.split(' ')
                col_name = code_tmp[0].strip()  # 分裂的列名
                operator = code_tmp[1].strip()  # 分裂的操作符

                if operator.find('<') > -1 or operator.find('>') > -1 or operator.find('=') > -1 or operator == 'in':
                    tree_code = single_code.replace(col_name, 'row["{colname}"]'.format(colname=col_name))
                elif operator == 'is':
                    tree_code = 'pd.isnull(row["{colname}"])'.format(colname=col_name)

                return tree_code

            if split_code_string.find(' or ') > -1:   # 多个条件的组合： or
                codes = split_code_string.split(' or ')
                tree_code = []
                for code_i in codes:
                    eval_i = '({})'.format(single_code_trans(code_i))
                    tree_code.append(eval_i)
                tree_code = ' or '.join(tree_code)
            elif split_code_string.find(' and ') > -1:   # 多个条件的组合： and
                codes = split_code_string.split(' and ')
                tree_code = []
                for code_i in codes:
                    eval_i = '({})'.format(single_code_trans(code_i))
                    tree_code.append(eval_i)
                tree_code = ' and '.join(tree_code)
            else:  # 单个条件
                tree_code = single_code_trans(split_code_string)

            return tree_code

        self.__grow_split_str_ = split_code_string
        code_eval = analysis_code_string()
        logic_sr = eval(code_eval)
        tree_code = to_tree_code()
        left_id, right_id = cal_next_level_id()

        logic_left = logic_sr & self.__node_logic
        self.left = self.__born_child(level=self.level_ + 1, node_logic=logic_left, split_code=tree_code, node_id=left_id)

        logic_right = (~logic_sr) & self.__node_logic
        self.right = self.__born_child(level=self.level_ + 1, node_logic=logic_right, split_code=None, node_id=right_id)

        self.__get_root().plot_tree()

    def __get_root(self):
        """获取根节点"""
        if self.__father is None:
            return self
        else:
            return self.__father.__get_root()

    def generate_tree_code(self):
        """生成决策树的代码"""
        # 本层的代码
        if self.level_ == 0:   # 根节点的代码
            self_level_code = """def tree(row):\n"""
        else:                  # 中间层节点的代码
            if self.__split_tree_code is not None:   # 左边的节点
                node_code = 'if {}: \n'.format(self.__split_tree_code)
            else:                               # 右边的节点
                node_code = 'else: \n'
            self_level_code = '    ' * self.level_ + node_code

        if self.is_leaf():   # 叶节点的代码（比中间节点多出一行赋值操作）
            self_level_code += '    ' * (self.level_ + 1) + 'node = "TreeNode_{}" \n'.format(self.node_id_)

        # 下一层的代码
        if not self.is_leaf():  # 若不是叶节点，则会有左、右子树
            self_level_code += self.left.generate_tree_code()
            self_level_code += self.right.generate_tree_code()

        # 加上最后的return
        if self.level_ == 0:
            self_level_code += '    return node'
            print(self_level_code)
        else:
            return self_level_code

    def is_leaf(self):
        """是否是叶节点"""
        return self.left is None

    def get_node(self, node_id):
        """获取指定 node_id 的那个树节点对象. 当树生长得较深时，通过此方法能快速获取到目标节点。若返回 -1 表示未找到指定的节点。

        参数:
        -----------
        node_id: int, 树节点的编号

        返回:
        -----------
        InteractTree 对象，编号等于参数 node_id 的那个节点"""
        # 本层的代码
        if self.node_id_ == node_id:
            return self
        elif self.is_leaf():
            return -1
        else:  # 下层代码
            result_left = self.left.get_node(node_id)   # result 只可能是 node 或 -1
            if result_left == -1:
                return self.right.get_node(node_id)
            else:
                return result_left

    def cal_woe_df(self, name=None):
        """汇总叶节点的信息，并计算 woe_df 。在根节点调用才有意义，其他节点调用时返回的是局部结果。
        name: 指定决策树的名字"""
        woe_list = []
        if self.is_leaf():
            woe_list.append(self.node_info_)
        else:
            woe_list.extend(self.left.cal_woe_df())
            woe_list.extend(self.right.cal_woe_df())
        if self.level_ == 0:
            woe_df = pd.DataFrame(woe_list)
            woe_df['IV'] = woe_df['IV_i'].sum()
            woe_df = woe_df.set_index('node_id')
            woe_df['colName'] = 'Tree' if name is None else name
            bad = (woe_df['All'] * woe_df['badRate']).map(lambda x: int(round(x, 0)))
            woe_df[0] = woe_df['All'] - bad
            woe_df[1] = bad
            woe_df = woe_df.sort_values('badRate', ascending=False)
            return woe_df[['colName', 0, 1, 'All', 'binPct', 'badRate', 'woe', 'IV_i', 'IV']]
        else:
            return woe_list

    def plot_tree(self):
        """画出决策树。只有根节点调用此方法，画出的才是完整的树。其余节点调用此方法，只返回局部子树的 dot 代码.
        所画的图保存路径： root.tree_img_path
        本方法需要调用 Graphviz，需要先安装该软件。
        如果画图失败，打开命令窗，以管理员权限执行以下命令即可：
        dot -c
        """

        def self_node_name():
            """本层的节点名"""
            return 'node_{}'.format(self.node_id_)

        def father_node_name():
            """计算父节点的节点名。 节点的命名规则是 node_{node_id_}"""
            node_id = bin(self.node_id_)
            father_id = eval(node_id[:-1])
            return 'node_{}'.format(father_id)

        def node_code(node):
            """生成节点的 dot 代码"""
            def label_str():
                """生成节点的label的字符串"""
                def pretty(code):
                    """把分割代码，转换成更精简的形式，以便显示在图上.
                    函数假设每棵树仅使用单个变量分割，缺失值要么在左边，要么在右边. """
                    if code.find(' or ') > -1:  # 函数假设：若是两个判断条件, 则必包含了缺失值
                        code1, code2 = code.split(' or ')
                        code1 = pretty(code1)
                        code2 = pretty(code2)
                        if code1.find('is nan') > -1:
                            return code2 + ' or nan'
                        else:
                            return code1 + ' or nan'
                    else:  # 单个分割代码的处理
                        if code.find(' in ') > -1 and len(code) > 40:  # 分割文本过长
                            code = ' in \n'.join(code.split(' in '))
                        return code

                label = "node_{node.node_id:.0f} (N={node.All:.0f})\n" \
                        "bad={node.badRate:0.4f}, pct={node.binPct:0.4f}\n" \
                        "woe={node.woe:0.4f}, ivi={node.IV_i:0.4f}\n".format(node=self.node_info_)
                # 加上分割代码
                if not self.is_leaf():
                    split_code = pretty(self.__grow_split_str_)
                    label += split_code
                return label

            node_name = self_node_name()
            label = label_str()
            color = 'green' if self.is_leaf() else 'lightblue'
            node_dot = '{name} [label="{label}", fillcolor={color}];\n'.format(label=label, name=node_name, color=color)
            return node_dot

        def connect_code_fun(name1, name2):
            """生成连接两个节点的 dot 代码"""
            if self.node_id_ == 2:
                code = '{0} -> {1} [labeldistance=2.5, labelangle=45, headlabel="True"];\n'.format(name1, name2)
            elif self.node_id_ == 3:
                code = '{0} -> {1} [labeldistance=2.5, labelangle=-45, headlabel="False"];\n'.format(name1, name2)
            else:
                code = '{0} -> {1} ;\n'.format(name1, name2)
            return code

        # 本层的 dot 代码
        if self.level_ == 0:  # 顶层的 dot 代码
            dot_code = 'digraph tree {\n' \
                       'node [shape=box, style="filled, rounded", color="black", fontname="SimSun"] ;\n' \
                       'edge [fontname="SimSun"] ;\n'   # 公共属性的声明，fontname的声明解决中文乱码问题
            dot_code += node_code(self)
        else:   # 其他层的 dot 代码，处理方式与顶层不同
            self_node = node_code(self)
            connect_code = connect_code_fun(father_node_name(), self_node_name())
            dot_code = self_node + connect_code    # 迭代式的，本层的代码生成，不需要依赖下层

        # 拼接下一层的 dot 代码
        if not self.is_leaf():
            dot_code += self.left.plot_tree()
            dot_code += self.right.plot_tree()

        # 拼完所有层代码之后，加上右花括号, 并输出图片
        if self.level_ == 0:
            dot_code += '}'
            img_path = self.tree_img_path_
            with open(img_path+'.dot', 'w', encoding='utf8') as f:
                f.write(dot_code)

            os.system('dot -Tpng {file}.dot -Gdpi=300 -o {file}.png'.format(file=img_path))
            os.remove(img_path+'.dot')

            # from graphviz import Source
            # from matplotlib import image
            #
            # _, ax = plt.subplots(1, 1)
            # g = Source(tree)
            # s = BytesIO()
            # s.write(g.pip(format='png'))
            # s.seek(0)
            # img = image.imread(s)
            #
            # ax.imshow(img)
            # ax.axis('off')
        else:  # 不是顶层的，都需要返回拼接代码给顶层
            return dot_code

    def split_rank(self):
        """把各个节点的最优分割信息汇总，并按降序排列。"""
        split_df = pd.DataFrame()
        if self.is_leaf():
            split_df = self.split_info_
            split_df['node_id'] = self.node_id_
        else:
            split_df = pd.concat([self.left.split_rank(), self.right.split_rank()])
        split_df = split_df.sort_values('gain_' + self.__get_root().criteria_, ascending=False)
        return split_df

    def cut_leaf(self, node_id):
        """剪掉指定 node_id 的叶节点。无返回值，会修改树本身。

        参数:
        ---------
        node_id: int, 要剪掉的叶节点的 node_id，剪掉此叶节点，亦会剪掉它的兄弟叶节点"""
        father_id = node_id // 2
        node = self.get_node(father_id)
        node.left = None
        node.right = None

        self.__get_root().plot_tree()


