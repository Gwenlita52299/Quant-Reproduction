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

def read_changepoint_results_and_fillna(
        file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .ffill()
        .dropna()
        .assign(
            cp_location_norm = lambda row:(row['t'] - row['cp_location'])
            / lookback_window_length
        )
    )

def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    return pd.concat(
        [
            read_changepoint_results_and_fillna(
                os.path.join(folder_path,f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )

def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    df_asset = df_asset[
        ~df_asset['close'].isna()
        | ~df_asset['close'].isnull()
        | (df_asset['close'] > 1e-8)
    ].copy()

    df_asset['srs'] = df_asset['close']
    ewm = df_asset['srs'].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset['srs'] = np.minimum(df_asset['srs'], means + VOL_THRESHOLD * stds)
    df_asset['srs'] = np.maximum(df_asset['srs'], means - VOL_THRESHOLD * stds)

    df_asset['daily_returns'] = cal_returns(df_asset['srs'])
    df_asset['daily_vol'] = cal_daily_vol(df_asset['daily_returns'])
    df_asset['target_returns'] = cal_vol_scaled_returns(
        df_asset['daily_returns'], df_asset['daily_vol']
    ).shift(-1)

    def cal_normalised_returns(day_offset):
        return (
            cal_returns(df_asset['srs'], day_offset)
            / df_asset['daily_vol']
            / np.sqrt(day_offset)
        )
    
    df_asset['norm_daily_return'] = cal_normalised_returns(1)
    df_asset['norm_monthly_return'] = cal_normalised_returns(21)
    df_asset['norm_quarterly_return'] = cal_normalised_returns(63)
    df_asset['norm_biannual_return'] = cal_normalised_returns(126)
    df_asset['norm_annual_return'] = cal_normalised_returns(252)

    trend_combinations = [(8,24),(16,48),(32,96)]
    for short_window, long_window in trend_combinations:
        df_asset[f'macd_{short_window}_{long_window}'] = MACDStrategy.cal_signal(
            df_asset['srs'], short_window, long_window
        )

    if len(df_asset):
        df_asset['day_of_week'] = df_asset.index.dayofweek
        df_asset['day_of_month'] = df_asset.index.day
        df_asset['week_of_year'] = df_asset.index.isocalendar().week.astype(int)
        df_asset['month_of_year'] = df_asset.index.month
        df_asset['year'] = df_asset.index.year
        df_asset['date'] = df_asset.index
    else:
        df_asset['day_of_week'] = []
        df_asset['day_of_month'] = []
        df_asset['week_of_year'] = []
        df_asset['month_of_year'] = []
        df_asset['year'] = []
        df_asset['date'] = []

    return df_asset.dropna()

def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
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