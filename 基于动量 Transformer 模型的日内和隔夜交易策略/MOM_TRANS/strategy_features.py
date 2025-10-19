import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

from empyrical import(
    sharpe_ratio, #夏普比率
    calmar_ratio, #卡玛比率
    sortino_ratio, #索提诺比率
    max_drawdown, #最大回撤
    downside_risk, #下行风险
    annual_return, #年化收益率
    annual_volatility #年化波动率
)

VOL_LOOKBACK = 60
VOL_TARGET = 0.15 
'''
原文件中给定的窗口期和波动率目标
'''

def cal_performance_metrics(data: pd.DataFrame, metric_suffix="", num_identifiers = None) -> dict:
    """计算策略指标

    参数：
        日收益率(pd.DataFrame):index为日期的日收益率DataFrame

    输出：
        dict：各项指标的字典
    """
    if not num_identifiers:
        num_identifiers = len(data.dropna()["identifier"].unique()) #计算有多少个标的
    ars = data.dropna().groupby(level=0)['captured_returns'].sum()/num_identifiers #ars(average returns)是按日期合并收益率后取品种的平均收益率
    return{
        f'annual_return{metric_suffix}': annual_return(ars),
        f'annual_volatility{metric_suffix}': annual_volatility(ars),
        f'sharpe_ratio{metric_suffix}': sharpe_ratio(ars),
        f'downside_risk{metric_suffix}': downside_risk(ars),
        f'sortino_ratio{metric_suffix}': sortino_ratio(ars),
        f'max_drawdown{metric_suffix}': -max_drawdown(ars),
        f'calmar_ratio{metric_suffix}': calmar_ratio(ars),
        f'win_rate{metric_suffix}': len(ars[ars>0.0]) / len(ars), #原代码用perc_pos_return来表达，但计算公式看用胜率更贴切
        f'profit_loss_ratio{metric_suffix}': np.mean(ars[ars>0.0]) / np.mean(np.abs(ars[ars<0.0])),
    }

def cal_performance_metrics_subset(ars: pd.DataFrame, metric_suffix="") -> dict: #只对已有组合收益率做少数指标
    """计算策略指标

    参数：
        平均日收益率(pd.DataFrame):index为日期的日收益率DataFrame

    输出：
        dict：各项指标的字典
    """
    return{
        f'annual_return{metric_suffix}': annual_return(ars),
        f'annual_volatility{metric_suffix}': annual_volatility(ars),
        f'downside_risk{metric_suffix}': downside_risk(ars),
        f'max_drawdown{metric_suffix}': -max_drawdown(ars),
    }

def cal_net_returns(data: pd.DataFrame, list_basis_points: List[float], identifiers = None):
    if not identifiers:
        identifiers = data['identifier'].unique().tolist()
    cost = np.atleast_2d(list_basis_points) * 1e-4 #将交易费用bps转换成小数的形式

    dfs = []
    for i in identifiers:
        data_slice = data[data['identifier'] == i].reset_index(drop=True)
        annualised_vol = data_slice['daily_vol']*np.sqrt(252)
        scaled_position = VOL_TARGET*data_slice['position']/annualised_vol #根据原先设定的年化波动率调整仓位
        transaction_costs = scaled_position.diff().abs().fillna(0.0).to_frame().to_numpy()*cost #初始减仓成本简化为0
        net_captured_returns = data_slice[['captured_returns']].to_numpy()-transaction_costs #计算扣除交易成本的净收益
        columns = list(map(lambda c: 'captured_returns_' + str(c).replace('.','_')+ '_bps',list_basis_points)) #把不同成本下的收益列命名
        dfs.append(pd.concat([data_slice, pd.DataFrame(net_captured_returns,columns=columns)],axis=1))
    return pd.concat(dfs).reset_index(drop=True)

def cal_sharpe_by_year(data: pd.DataFrame, suffix: str = None) -> dict:
    """用dataframe计算每年的夏普比率

    参数:
        日收益率(pd.DataFrame):index为日期的日收益率DataFrame

    输出:
        dict: 年化夏普比率的字典
    """ 
    if not suffix:
        suffix = ''
    
    data = data.copy()
    data['year'] = data.index.year

    sharpes = (
        data.dropna()[['year','captured_returns']]
        .groupby(level=0)
        .mean() #按日期分组取日均值
        .groupby('year')
        .apply(lambda y: sharpe_ratio(y['captured_returns'])) #按年份分组计算 Sharpe ratio
    )

    sharpes.index = 'sharpe_ratio_' + shapres.index.map(int).map(str) + suffix

    return sharpes.to_dict()

def cal_returns(ps: pd.Series, day_offset: int = 1) -> pd.Series:
    """计算历史收益率，时间窗口由day_offset决定

    参数:
        ps (pd.Series): 价格的时间序列
        day_offset (int, optional): 用于计算收益率的天数，默认为1

    输出:
        pd.Series: 收益率的时间序列
    """    
    returns = ps / ps.shift(day_offset) - 1.0
    return returns

def cal_daily_vol(daily_returns):
    return(
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK) #使用指数加权移动平均，使其对近期收益权重更大，更好捕捉波动率变化
        .std()
        .bfill()) #前VOL_LOOKBACK-1项缺失值，故用后向填充


def cal_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
    if not len(daily_vol):
        daily_vol = cal_daily_vol(daily_returns) #若没传每日波动率数据，则调用函数计算
    annualised_vol = daily_vol * np.sqrt(252)
    return daily_returns * VOL_TARGET / annualised_vol.shift(1) #用前一天的波动率数据计算避免未来函数

def cal_intermediate_trend_strategy(
        ps: pd.Series, w: float, voaltility_scaling=True
) -> pd.Series:
    """计算中期趋势指标

    参数:
        ps (pd.Series): 价格序列
        w (float): 权重 w=0 表示只用年度动量（Moskowitz TSMOM），w=1 表示只用月度动量。
        volatility_scaling (bool, optional):可选是否使用波动率缩放，默认使用

    输出:
        pd.Series: 收益率序列
    """
    daily_returns = cal_returns(ps)
    monthly_returns = cal_returns(ps, 21)
    annual_returns = cal_returns(ps, 252)

    next_day_returns = (
        cal_vol_scaled_returns(daily_returns).shift(-1) #将信号收益与信号出现对齐，并未使用未来数据
        if voaltility_scaling
        else daily_returns.shift(-1)
    )

    return (
        w * np.sign(monthly_returns) * next_day_returns
        + (1-w) * np.sign(annual_returns) * next_day_returns
    )

class MACDStrategy:
    def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
        """用于计算MACD的参数组合来源于该篇论文 https://arxiv.org/pdf/1904.04912.pdf

        参数:
            trend_combinations (List[Tuple[float, float]], optional): 长短期组合，默认为空
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def cal_signal(ps: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """通过长短周期计算MACD信号

        参数:
            ps ([type]): 价格序列
            short_timescale ([type]): 短周期
            long_timescale ([type]): 长周期

        输出:
            float: MACD信号
        """
        def cal_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)
        
        macd = (
            ps.ewm(halflife=cal_halflife(short_timescale)).mean()
            - ps.ewm(halflife=cal_halflife(long_timescale)).mean()
        )
        q = macd / ps.rolling(63).std().bfill() #价格标准化
        return q / q.rolling(252).std().bfill() #信号标准化
    
    @staticmethod
    def scale_signal(x):
        return x * np.exp(-(x**2)/4) / 0.89 #0.89应该是经验参数
    
    def cal_combined_signal(self, ps: pd.Series) -> float:
        """组合MACD信号

        参数:
            ps (pd.Series): 价格序列

        输出:
            float: MACD组合信号
        """
        return np.sum(
            [self.cal_signal(ps, S, L) for S, L in self.trend_combinations]
        ) / len(self.trend_combinations)