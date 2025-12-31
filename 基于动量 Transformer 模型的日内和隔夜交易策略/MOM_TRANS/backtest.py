import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
import shutil
import gc
import copy

import json

from MOM_TRANS.model_input import ModelFeatures
from MOM_TRANS.deep_momentum_network import LstmDeepMomentumNetworkModel as LSTMDMN
from MOM_TRANS.momentum_transformer import TftDeepMomentumNetworkModel
from MOM_TRANS.strategy_features import (
    VOL_TARGET,
    cal_performance_metrics,
    cal_performance_metrics_subset,
    cal_sharpe_by_year,
    cal_net_returns,
    annual_volatility,
)

from setting.default import BACKTEST_AVERAGE_BASIS_POINTS

from setting.hyperparameter_grid import HP_MINIBATCH_SIZE

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def _get_directory_name(
    experiment_name: str, train_interval: Tuple[int, int, int] = None
) -> str:
    if train_interval:
        # 使用年份后两位，例如 24-25 而不是 2024-2025
        y1 = str(train_interval[1])[-2:]
        y2 = str(train_interval[2])[-2:]
        return os.path.join('r', experiment_name, f'{y1}{y2}')  # r 代替 results
    else:
        return os.path.join('r', experiment_name)
    
def _basis_point_suffix(basis_points: float = None) -> str:
    if not basis_points:
        return ''
    return "_" + str(basis_points).replace('.','_') + '_bps'

def _interval_suffix(
    train_interval: Tuple[int, int, int], basis_points: float = None
) -> str:
    return f'_{train_interval[1]}_{train_interval[2]}' + _basis_point_suffix(
        basis_points
    )

def _results_from_all_windows(
    experiment_name: str, train_intervals: List[Tuple[int, int, int]]
):
    return pd.concat(
        [
            pd.read_json(
                os.path.join(
                    _get_directory_name(experiment_name, interval), 'results.json'
                ),
            )
            for interval in train_intervals
        ]
    )

def _get_asset_classes(asset_class_dictionary: Dict[str, str]):
    return np.unique(list(asset_class_dictionary.values())).tolist()

def _captured_returns_from_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    volatility_rescaling: bool = True,
    only_standard_windows: bool = True,
    volatilites_known: List[float] = None,
    filter_identifiers: List[str] = None,
    captured_returns_col: str = 'captured_returns',
    standard_window_size: int = 1,
) -> pd.Series:
    srs_list = []
    volatilites = volatilites_known if volatilites_known else []
    for interval in train_intervals:
        if only_standard_windows and (
            interval[2] - interval[1] == standard_window_size
        ):
            df = pd.read_csv(
                os.path.join(
                    _get_directory_name(experiment_name, interval),
                    'captured_returns_sw.csv',
                ),
            )

            if filter_identifiers:
                filter = pd.DataFrame({'identifier': filter_identifiers})
                df = df.merge(filter, on='identifier')
            num_identifiers = len(df['identifier'].unique())
            srs = df.groupby('time')[captured_returns_col].sum() / num_identifiers
            srs_list.append(srs)
            if volatility_rescaling and not volatilites_known:
                volatilites.append(annual_volatility(srs))
    if volatility_rescaling:
        return pd.concat(srs_list) * VOL_TARGET / np.mean(volatilites)
    else:
        return pd.concat(srs_list)
    
def save_results(
    results_sw: pd.DataFrame,
    output_directory: str,
    train_interval: Tuple[int, int, int],
    num_identifiers: int,
    asset_class_dictionary: Dict[str,str],
    extra_metrics: dict={},
):
    asset_classes = ['ALL']
    results_asset_class = [results_sw]
    if asset_class_dictionary:
        results_sw['asset_class'] = results_sw['identifier'].map(
            lambda i: asset_class_dictionary[i]
        )
        classes  = _get_asset_classes(asset_class_dictionary)
        for ac in classes:
            results_asset_class += [results_sw[results_sw['asset_class'] == ac]]
        asset_classes += classes

    metrics = {}
    for ac, results_ac in zip(asset_classes, results_asset_class):
        suffix = _interval_suffix(train_interval)
        if ac == 'ALL' and extra_metrics:
            ac_metrics = extra_metrics.copy()
        else:
            ac_metrics = {}
        for basis_points in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _interval_suffix(train_interval, basis_points)
            if basis_points:
                results_ac_bps = results_ac.drop(columns='captured_returns').rename(
                    columns={
                        'captured_returns'
                        + _basis_point_suffix(basis_points): 'captured_returns'
                    }
                )
            else:
                results_ac_bps = results_ac

            ac_metrics = {
                **ac_metrics,
                **cal_performance_metrics(
                    results_ac_bps.set_index('time'), suffix, num_identifiers
                ),
                **cal_sharpe_by_year(
                    results_ac_bps.set_index('time'), _basis_point_suffix(basis_points)
                ),
            }
        metrics = {**metrics, ac: ac_metrics}

    with open(os.path.join(output_directory, 'results.json'), 'w') as file:
        file.write(json.dumps(metrics, indent=4))

def aggregate_and_save_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    asset_class_dictionary: Dict[str, str],
    standard_window_size: int,
):
    """聚合所有时间窗口的结果并保存
    
    参数：
        experiment_name: 实验名称
        train_intervals: 训练时间窗口列表
        asset_class_dictionary: 资产类别映射字典
        standard_window_size: 标准窗口大小（年）
    """
    directory = _get_directory_name(experiment_name)
    all_results = _results_from_all_windows(experiment_name, train_intervals)

    # 定义指标列表
    _metrics = [
        'annual_return',
        'annual_volatility',
        'sharpe_ratio',
        'downside_risk',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'win_rate',  # 已从 perc_pos_return 改为 win_rate
        'profit_loss_ratio',
    ]
    _rescaled_metrics = [
        'annual_return_rescaled',
        'annual_volatility_rescaled',
        'downside_risk_rescaled',
        'max_drawdown_rescaled',
    ]

    # 初始化指标列表
    metrics = []
    rescaled_metrics = []  # 注意：无下划线

    # 为每个基点生成完整的指标名称
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        suffix = _basis_point_suffix(bp)
        metrics += list(map(lambda m: m + suffix, _metrics))
        rescaled_metrics += list(map(lambda m: m + suffix, _rescaled_metrics))

    # 确定资产类别
    if asset_class_dictionary:
        asset_classes = ['ALL'] + _get_asset_classes(asset_class_dictionary)
    else:
        asset_classes = ['ALL']

    average_metrics = {}
    list_metrics = {}

    # 创建资产类别到 ticker 的映射
    asset_class_tickers = (
        pd.DataFrame.from_dict(asset_class_dictionary, orient='index')
        .reset_index()
        .set_index(0)
        if asset_class_dictionary
        else None
    )

    # 对每个资产类别计算指标
    for asset_class in asset_classes:
        # 初始化结果字典
        average_results = dict(
            zip(
                metrics + rescaled_metrics,
                [[] for _ in range(len(metrics + rescaled_metrics))],
            )
        )
        
        asset_results = all_results[asset_class]

        # 为每个基点初始化年度 Sharpe ratio 列表
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            average_results[f'sharpe_ratio_years{suffix}'] = []

        # 遍历所有时间窗口
        for interval in train_intervals:
            # 只处理标准窗口
            if interval[2] - interval[1] == standard_window_size:
                # 收集所有指标
                for m in _metrics:
                    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                        suffix = _interval_suffix(interval, bp)
                        metric_key = m + _basis_point_suffix(bp)
                        lookup_key = m + suffix
                        
                        # 安全读取指标
                        try:
                            if lookup_key in asset_results:
                                value = asset_results[lookup_key]
                                # 确保值不是 NaN 字符串
                                if isinstance(value, str) and value.lower() == 'nan':
                                    value = np.nan
                                average_results[metric_key].append(value)
                            else:
                                print(f"⚠️  缺失指标: {lookup_key}，使用 NaN")
                                average_results[metric_key].append(np.nan)
                        except Exception as e:
                            print(f"⚠️  读取指标 {lookup_key} 失败: {e}")
                            average_results[metric_key].append(np.nan)

            # 收集年度 Sharpe ratio
            for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                suffix = _basis_point_suffix(bp)
                for year in range(interval[1], interval[2]):
                    year_key = f'sharpe_ratio_{int(year)}{suffix}'
                    years_key = f'sharpe_ratio_years{suffix}'
                    
                    try:
                        # 确保键存在
                        if years_key not in average_results:
                            average_results[years_key] = []
                        
                        # 读取年度 Sharpe
                        if year_key in asset_results:
                            value = asset_results[year_key]
                            if isinstance(value, str) and value.lower() == 'nan':
                                value = np.nan
                            average_results[years_key].append(value)
                        else:
                            print(f"⚠️  缺失年度指标: {year_key}，使用 NaN")
                            average_results[years_key].append(np.nan)
                    except Exception as e:
                        print(f"⚠️  读取年度指标 {year_key} 失败: {e}")
                        if years_key not in average_results:
                            average_results[years_key] = []
                        average_results[years_key].append(np.nan)

        # 计算重缩放指标
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            
            try:
                # 获取波动率列表
                vol_key = 'annual_volatility' + suffix
                volatilites_known = average_results.get(vol_key, [])
                
                # 获取所有窗口的收益率
                all_captured_returns = _captured_returns_from_all_windows(
                    experiment_name,
                    train_intervals,
                    volatility_rescaling=True,
                    only_standard_windows=True,
                    volatilites_known=volatilites_known,
                    filter_identifiers=(
                        None
                        if asset_class == 'ALL'
                        else asset_class_tickers.loc[
                            asset_class, asset_class_tickers.columns[0]
                        ].tolist()
                        if asset_class_tickers is not None
                        else None
                    ),
                    captured_returns_col=f'captured_returns{suffix}',
                )
                
                yrs = pd.to_datetime(all_captured_returns.index).year
                
                # 对每个标准窗口计算重缩放指标
                for interval in train_intervals:
                    if interval[2] - interval[1] == standard_window_size:
                        srs = all_captured_returns[
                            (yrs >= interval[1]) & (yrs < interval[2])
                        ]
                        
                        if len(srs) > 0:
                            rescaled_dict = cal_performance_metrics_subset(
                                srs, f'_rescaled{suffix}'
                            )
                            
                            for m in _rescaled_metrics:
                                metric_key = m + suffix
                                if metric_key in rescaled_dict:
                                    average_results[metric_key].append(rescaled_dict[metric_key])
                                else:
                                    print(f"⚠️  缺失重缩放指标: {metric_key}，使用 NaN")
                                    average_results[metric_key].append(np.nan)
                        else:
                            print(f"⚠️  窗口 {interval} 没有数据")
                            for m in _rescaled_metrics:
                                metric_key = m + suffix
                                average_results[metric_key].append(np.nan)
                                
            except Exception as e:
                print(f"⚠️  处理重缩放指标失败 (bp={bp}): {e}")
                for m in _rescaled_metrics:
                    metric_key = m + suffix
                    if metric_key not in average_results:
                        average_results[metric_key] = []
                    # 为每个标准窗口添加 NaN
                    standard_windows = [i for i in train_intervals if i[2] - i[1] == standard_window_size]
                    for _ in standard_windows:
                        average_results[metric_key].append(np.nan)

        # 保存窗口历史（用于计算标准差）
        window_history = copy.deepcopy(average_results)
        
        # 计算平均值（过滤 NaN）
        for key in average_results:
            values = average_results[key]
            # 过滤 NaN 值
            valid_values = [
                v for v in values 
                if not (isinstance(v, float) and np.isnan(v))
            ]
            
            if valid_values:
                average_results[key] = np.mean(valid_values)
            else:
                average_results[key] = np.nan
                print(f"⚠️  指标 {key} 所有值都是 NaN")

        # 计算年度 Sharpe ratio 的标准差
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            years_key = f'sharpe_ratio_years{suffix}'
            std_key = f'sharpe_ratio_years_std{suffix}'
            
            if years_key in window_history:
                values = window_history[years_key]
                valid_values = [
                    v for v in values 
                    if not (isinstance(v, float) and np.isnan(v))
                ]
                
                if valid_values:
                    average_results[std_key] = np.std(valid_values)
                else:
                    average_results[std_key] = np.nan
            else:
                print(f"⚠️  缺失 {years_key}，标准差设为 NaN")
                average_results[std_key] = np.nan

        # 保存到总结果中
        average_metrics = {**average_metrics, asset_class: average_results}
        list_metrics = {**list_metrics, asset_class: window_history}

    # 保存到文件（处理 NaN 和特殊值）
    try:
        with open(os.path.join(directory, 'average_results.json'), 'w') as file:
            file.write(json.dumps(average_metrics, indent=4, default=str))
        print(f"✓ 平均结果已保存到: {os.path.join(directory, 'average_results.json')}")
    except Exception as e:
        print(f"⚠️  保存平均结果失败: {e}")
    
    try:
        with open(os.path.join(directory, 'list_results.json'), 'w') as file:
            file.write(json.dumps(list_metrics, indent=4, default=str))
        print(f"✓ 列表结果已保存到: {os.path.join(directory, 'list_results.json')}")
    except Exception as e:
        print(f"⚠️  保存列表结果失败: {e}")
    
    print(f"✓ 聚合完成: {experiment_name}")


def run_single_window(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    changepoint_lbws: List[int],
    skip_if_completed: bool = True,
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = HP_MINIBATCH_SIZE,
):
    directory = _get_directory_name(experiment_name, train_interval)

    if skip_if_completed and os.path.exists(os.path.join(directory, 'results.json')):
        print(
            f'Skipping {train_interval[1]}-{train_interval[2]} because already completed.'
        )
        return
    
    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data['date'] = raw_data['date'].astype('datetime64[ns]')

    model_features = ModelFeatures(
        raw_data,
        params['total_time_steps'],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params['split_tickers_individually'],
        train_valid_ratio=params['train_valid_ratio'],
        add_ticker_as_static=(params['architecture'] == 'TFT'),
        time_features=params['time_features'],
        lags=params['force_output_sharpe_length'],
        asset_class_dictionary=asset_class_dictionary,
    )

    hp_directory = os.path.join(directory, 'hp')

    if params['architecture'] == 'LSTM':
        dmn = LSTMDMN(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
        )
    elif params['architecture'] == 'TFT':
        dmn = TftDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
            **{
                'column_definition': model_features.get_column_definition(),
                'num_encoder_steps': 0,
                'stack_size': 1,
                'num_heads': 4,
            },
        )
    else:
        dmn = None
        raise Exception(f"{params['architecture']} is not a valid architecture.")

    best_hp, best_model = dmn.hyperparameter_search(
        model_features.train, model_features.valid
    )
    val_loss = dmn.evaluate(model_features.valid, best_model)

    print(f'Best validation loss = {val_loss}')
    print(f'Best params:')
    for k in best_hp:
        print(f'{k} = {best_hp[k]}')

    with open(os.path.join(directory, 'best_hyperparameters.json'), 'w') as file:
        file.write(json.dumps(best_hp))

    print('Predicting on test set...')

    results_sw, performance_sw = dmn.get_positions(
        model_features.test_sliding,
        best_model,
        sliding_window=True,
        years_geq=train_interval[1],  # ← 改为 years_geq
        years_lt=train_interval[2],   # ← 改为 years_lt
    )
    print(f'performance(sliding window) = {performance_sw}')

    results_sw = results_sw.merge(
        raw_data.reset_index()[['ticker','date','daily_vol']].rename(
            columns={'ticker': 'identifier', 'date': 'time'}
        ),
        on=['identifier','time'],
    )
    results_sw = cal_net_returns(
        results_sw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_sw.to_csv(os.path.join(directory, 'captured_returns_sw.csv'))

    results_fw, performance_fw = dmn.get_positions(
        model_features.test_fixed,
        best_model,
        sliding_window=False,
        years_geq=train_interval[1],  # ← 改为 years_geq
        years_lt=train_interval[2],   # ← 改为 years_lt
    )
    print(f'performance (fixed window) = {performance_fw}')
    results_fw = results_fw.merge(
        raw_data.reset_index()[['ticker','date','daily_vol']].rename(
            columns={'ticker': 'identifier', 'date': 'time'}
        ),
        on=['identifier','time'],
    )
    results_fw = cal_net_returns(
        results_fw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_fw.to_csv(os.path.join(directory, 'captured_returns_sw.csv'))

    with open(os.path.join(directory, "fixed_params.json"), "w") as file:
        file.write(
            json.dumps(
                dict(
                    **params,
                    **model_features.input_params,
                    **{
                        "changepoint_lbws": changepoint_lbws
                        if changepoint_lbws
                        else [],
                        "features_file_path": features_file_path,
                    },
                ),
                indent=4,
            )
        )

    best_directory = os.path.join(directory, 'best')
    best_model.save_weights(os.path.join(best_directory, 'checkpoints', 'checkpoints'))
    with open(os.path.join(best_directory,'hyperparameters.json'), 'w') as file:
        file.write(json.dumps(best_hp, indent=4))
    shutil.rmtree(hp_directory)
    
    save_results(
        results_sw,
        directory,
        train_interval,
        model_features.num_tickers,
        asset_class_dictionary,
        {
            'performance_sw': performance_sw,
            'performance_fw': performance_fw,
            'val_loss': val_loss,
        },
    )

    del best_model
    gc.collect()
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    
def run_all_windows(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
    hp_minibatch_size=HP_MINIBATCH_SIZE,
    standard_window_size=1,
):
    for interval in train_intervals:
        run_single_window(
            experiment_name,
            features_file_path,
            interval,
            params,
            changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
            hp_minibatch_size=hp_minibatch_size,
        )

    aggregate_and_save_all_windows(
        experiment_name, train_intervals, asset_class_dictionary, standard_window_size
    )

def intermediate_momentum_position(w: float, returns_data: pd.DataFrame) -> pd.Series:
    return w * np.sign(returns_data['norm_monthly_return']) + (1-w) * np.sign(
        returns_data['norm_annual_return']
    )

def run_classical_methods(
    features_file_path,
    train_intervals,
    reference_experiment,
    long_only_experimengt_name='long_only',
    tsmom_experiment_name='tsmom',
):
    directory = _get_directory_name(long_only_experimengt_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = _get_directory_name(tsmom_experiment_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for train_interval in train_intervals:
        directory = _get_directory_name(tsmom_experiment_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        raw_data = pd.read_csv(features_file_path, parse_dates=True)
        reference = pd.read_csv(
            f'results/{reference_experiment}/{train_interval[1]}-{train_interval[2]}/captured_returns_sw.csv',
            parse_dates=True,
        )
        returns_data = raw_data.merge(
            reference[['time','identifier','returns']],
            left_on=['date','ticker'],
            right_on=['time','identifier'],
        )
        returns_data['position'] = intermediate_momentum_position(0, returns_data)
        returns_data['captured_returns'] = (
            returns_data['position'] * returns_data['returns']
        )
        returns_data = returns_data.reset_index()[
            ['identifier','time','returns','position','captured_returns']
        ]
        returns_data.to_csv(f'{directory}/captured_returns_sw')

        directory = _get_directory_name(long_only_experimengt_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        returns_data['position'] = 1.0
        returns_data['captured_returns'] = (
            returns_data['position'] * returns_data['returns']
        )
        returns_data.to_csv(f'{directory}/captured_returns_sw.csv')