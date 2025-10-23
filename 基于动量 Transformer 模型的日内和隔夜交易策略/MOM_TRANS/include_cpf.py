import os
import numpy as np
import pandas as pd

from MOM_TRANS.strategy_features import(
    MACDStrategy,
    cal_returns,
    cal_daily_vol,
    cal_vol_scaled_returns,
)

VOL_THRESHOLD = 5
HALFLIFE_WINSORISE = 252

def read_cpd_results_and_fillna(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """读取外部拐点检测模块的数据，对缺失值使用前向填充


    参数:
        file_path (str): csv文件所在地址
        lookback_window_length (int): 填补规范位置的空白所必需的

    输出:
        pd.DataFrame: 包含拐点信息的dataframe
    """
    return(
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .ffill()
        .dropna() #如果第一个值是na
        .assign(
            cp_location_norm = lambda row: (row['t'] - row['cp_location'])
            / lookback_window_length
        )
    )

def prepare_cpd_features(
        folder_path:str, lookback_window_length:int
) -> pd.DataFrame:
    """读取所有资产的cpd模块检测结果


    参数:
        file_path (str): 文档所在路径
        lookback_window_length (int): 回测窗口期

    输出:
        pd.DataFrame: 所有标的的拐点得分和位置
    """
    return pd.concat(
        [
            read_cpd_results_and_fillna(
                os.path.join(folder_path,f),lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )

def momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """为深度学习准备输入特征

    参数:
        df_asset (pd.DataFrame): 标的的时序数据，包含收盘价一列

    输出:
        pd.DataFrame: 输入的特征
    """
    df_asset = df_asset[
        ~df_asset['close'].isna()
        | ~df_asset['close'].isnull()
        |(df_asset['close']>1e-8)
    ].copy() #去除异常行

    #去除极端值
    df_asset['ars']=df_asset['close']
    ewm = df_asset['ars'].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset['ars'] = np.minimum(df_asset['ars'], means + VOL_THRESHOLD * stds)
    df_asset['ars'] = np.maximum(df_asset['ars'], means-VOL_THRESHOLD * stds)

    df_asset['daily_returns'] = cal_returns(df_asset['ars'])
    df_asset['daily_vol'] = cal_daily_vol(df_asset['daily_returns'])

    #波动率缩放并对齐日期
    df_asset['target_returns'] = cal_vol_scaled_returns(
        df_asset['daily_returns'], df_asset['daily_vol']
    ).shift(-1)

    def cal_normalised_returns(day_offset):
        return(
            cal_returns(df_asset['ars'], day_offset)
            /df_asset['daily_vol']
            /np.sqrt(day_offset)
        )

    df_asset['norm_daily_return'] = cal_normalised_returns(1)
    df_asset['norm_monthly_return'] = cal_normalised_returns(21)
    df_asset['norm_quarterly_return'] = cal_normalised_returns(63)
    df_asset['norm_biannual_return'] = cal_normalised_returns(126)
    df_asset['norm_annual_return'] = cal_normalised_returns(252)

    trend_combinations = [(8,24),(16,48),(32,96)]
    for short_window, long_window in trend_combinations:
        df_asset[f'macd_{short_window}_{long_window}'] = MACDStrategy.cal_combined_signal(
            df_asset['ars'], short_window, long_window
        )

    #日期特征
    if len(df_asset):
        df_asset['day_of_week'] = df_asset.index.dayofweek
        df_asset['day_of_month'] = df_asset.index.day
        df_asset['day_of_year'] = df_asset.index.weekofyear
        df_asset['month_of_year'] = df_asset.index.month
        df_asset['year'] = df_asset.index.year
        df_asset['date'] = df_asset.index
    else:
        df_asset['day_of_week'] = []
        df_asset['day_of_month'] = []
        df_asset['day_of_year'] = []
        df_asset['month_of_year'] = []
        df_asset['year'] = []
        df_asset['date'] = []

    return df_asset.dropna()

def include_cpd_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length:int
) -> pd.DataFrame:
    """将动量特征与拐点特征合并

    参数:
        features (pd.DataFrame): 特征
        cpd_folder_name (pd.DataFrame): 包含拐点检测的文档
        lookback_window_length (int): 拐点检测使用的窗口期

    输出:
        pd.DataFrame: 包含拐点检测得分和位置的特征
    """ 
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ['ticker','cp_location_norm','cp_score']
        ]
        .rename(
            columns={
                'cp_location_norm': f'cp_rl_{lookback_window_length}',
                'cp_score': f'cp_score_{lookback_window_length}',
            }
        )
        .reset_index(),
        on=['date','ticker'],
    )

    features.index = features['date']

    return features