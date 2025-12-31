import csv
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union
import os

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.kernels import ChangePoints, Matern32
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import bijectors as tfb

Kernel = gpflow.kernels.base.Kernel

MAX_ITERATIONS = 200 #Scipy优化器最大迭代次数

class ChangePointsWithBounds(ChangePoints):
    def __init__(
        self,
        kernels: Tuple[Kernel, Kernel],
        location: float,
        interval: Tuple[float, float],
        steepness: float = 1.0,
        name: Optional[str] = None,
    ):
        '''实现单一拐点检测，并把拐点位置限制在给定区间

        参数：
            kernels (Tuple[Kernel, Kernel]): 两个核（左/右）
            location (float): 拐点的初始位置
            interval (Tuple[float, float]): 拐点的区间范围
            steepness (float, optional): 梯度初始值，默认为1.0.
            name (Optional[str], optional): class name. Defaults to None.
        
        警告:
            ValueError: 如果初始拐点位置不在interval内则报错
        '''
        # overwrite the locations variable to enforce bounds
        if location < interval[0] or location > interval[1]:
            raise ValueError(
                "Location {loc} is not in range [{low},{high}]".format(
                    loc=location, low=interval[0], high=interval[1]
                )
            )
        locations = [location]
        super().__init__(
            kernels=kernels, locations=locations, steepness=steepness, name=name
        )

        affine = tfb.Shift(tf.cast(interval[0], tf.float64))(
            tfb.Scale(tf.cast(interval[1] - interval[0], tf.float64))
        )
        self.locations = gpflow.base.Parameter(
            locations, transform=tfb.Chain([affine, tfb.Sigmoid()]), dtype=tf.float64
        )
    
    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        locations = tf.reshape(self.locations, (1,1,-1))
        steepness = tf.reshape(self.steepness, (1,1,-1))
        return tf.sigmoid(steepness * (X[:,:,None] - locations))
    
def fit_matern_kernel(
    time_series_data: pd.DataFrame,
    variance: float = 1.0,
    lengthscale: float = 1.0,
    likelihood_variance: float = 1.0,
) -> Tuple[float,Dict[str, float]]:
    '''用Matern32核拟合时间序列

    参数:
        time_series_data (pd.DataFrame): 时间序列数据 X and Y
        variance (float, optional): 初始化方差。默认为 1.0。
        lengthscale (float, optional): 初始化缩放长度。默认为 1.0
        likelihood_variance (float, optional): 初始化似然方差。默认为 1.0。

    输出:
        Tuple[float, Dict[str, float]]: 拟合GP后的负对数边际似然和参数
    '''
    m = gpflow.models.GPR(
        data=(
            time_series_data.loc[:,['X']].to_numpy(),
            time_series_data.loc[:,['Y']].to_numpy(),
        ),
        kernel=Matern32(variance=variance, lengthscales=lengthscale),
        noise_variance=likelihood_variance,
    )
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=MAX_ITERATIONS)
    ).fun
    params = {
        'kM_variance': m.kernel.variance.numpy(),
        'kM_lengthscales': m.kernel.lengthscales.numpy(),
        'kM_likelihood_variance': m.likelihood.variance.numpy(),
    }
    return nlml, params

def fit_changepoint_kernel(
    time_series_data: pd.DataFrame,
    k1_variance: float = 1.0,
    k1_lengthscale: float = 1.0,
    k2_variance: float = 1.0,
    k2_lengthscale: float = 1.0,
    kC_likelihood_variance = 1.0,
    kC_changepoint_location = None,
    kC_steepness = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """在时间序列上拟合changepoint核

    参数:
        time_series_data (pd.DataFrame): 时间序列数据X和Y
        k1_variance (float, optional): 初始化方差k1。默认为 1.0。
        k1_lengthscale (float, optional): 初始化缩放长度 k1。默认为 1.0。
        k2_variance (float, optional): 初始化方差k2。默认为 1.0。
        k2_lengthscale (float, optional): 初始化缩放长度 k2。默认为 1.0。
        kC_likelihood_variance (float, optional): 初始化似然方差。默认为 1.0.
        kC_changepoint_location (float, optional): 初始化拐点位置，如果为空会计算取中点，默认为空.
        kC_steepness (float, optional): 初始化梯度。默认为1.0。

    输出:
        Tuple[float, float, Dict[str, float]]: 拐点位置, 拟合GP后的负对数边际似然和参数
    """ 
    if not kC_changepoint_location:
        kC_changepoint_location = (
            time_series_data['X'].iloc[0] + time_series_data['X'].iloc[-1]
        ) / 2.0

    m = gpflow.models.GPR(
        data = (
            time_series_data.loc[:,['X']].to_numpy(),
            time_series_data.loc[:,['Y']].to_numpy(),
        ),
        kernel = ChangePointsWithBounds(
            [
                Matern32(variance=k1_variance,lengthscales=k1_lengthscale),
                Matern32(variance=k2_variance,lengthscales=k2_lengthscale),
            ],
            location=kC_changepoint_location,
            interval=(time_series_data['X'].iloc[0],time_series_data['X'].iloc[-1]),
            steepness=kC_steepness,
        ),
    )
    m.likelihood.variance.assign(kC_likelihood_variance)
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=200)
    ).fun
    changepoint_location = m.kernel.locations[0].numpy()
    params = {
        'k1_variance': m.kernel.kernels[0].variance.numpy().flatten()[0],
        'k1_lengthscale': m.kernel.kernels[0].lengthscales.numpy().flatten()[0],
        'k2_varianve': m.kernel.kernels[1].variance.numpy().flatten()[0],
        'k2_lengthscale': m.kernel.kernels[1].lengthscales.numpy().flatten()[0],
        'kC_likelihood_variance': m.likelihood.variance.numpy().flatten()[0],
        'kC_changepoint_location': changepoint_location,
        'kC_steepness': m.kernel.steepness.numpy(),
    }
    return changepoint_location, nlml, params

def changepoint_severity(
    kC_nlml: Union[float, List[float]], kM_nlml: Union[float, List[float]]
) -> float:
    '''拐点得分 详细内容在 https://arxiv.org/pdf/2105.13727.pdf

    参数:
        kC_nlml (Union[float, List[float]]): 拟合拐点的负对数边际似然
        kM_nlml (Union[float, List[float]]): 拟合Matern32的负对数边际似然

    输出:
        float: 拐点得分   
    '''
    normalized_nlml = kC_nlml - kM_nlml
    return 1 - 1 / (np.mean(np.exp(-normalized_nlml)) + 1)

def changepoint_loc_and_score(
    time_series_data_window: pd.DataFrame,
    kM_variance: float = 1.0,
    kM_lengthscale: float = 1.0,
    kM_likelihood_variance: float = 1.0,
    k1_variance: float = None,
    k1_lengthscale: float = None,
    k2_variance: float = None,
    k2_lengthscale: float = None,
    kC_likelihood_variance = 1.0,
    kC_changepoint_location = None,
    kC_steepness = 1.0,
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
    """对单一窗口的时间序列计算拐点得分和位置

    参数:
        time_series_data_window (pd.DataFrame): time-series with columns X and Y
        kM_variance (float, optional): variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_lengthscale (float, optional): lengthscale initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_likelihood_variance (float, optional): likelihood variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        k1_variance (float, optional): variance initialisation for Changepoint kernel k1, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k1_lengthscale (float, optional): lengthscale initialisation for Changepoint kernel k1, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        k2_variance (float, optional): variance initialisation for Changepoint kernel k2, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k2_lengthscale (float, optional): lengthscale initialisation for for Changepoint kernel k2, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        kC_likelihood_variance ([type], optional): likelihood variance initialisation for Changepoint kernel. Defaults to None.
        kC_changepoint_location ([type], optional): changepoint location initialisation for Changepoint, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): changepoint location initialisation for Changepoint. Defaults to 1.0.

    输出:
        Tuple[float, float, float, Dict[str, float], Dict[str, float]]: 拐点得分，位置
        拐点位置归一化结果, Matern 3/2 核参数, 拐点核参数
    """
    time_series_data = time_series_data_window.copy()
    Y_data = time_series_data[['Y']].values
    time_series_data[['Y']] = StandardScaler().fit(Y_data).transform(Y_data) #对Y进行标准化

    try:
        (kM_nlml, kM_params) = fit_matern_kernel(
            time_series_data, kM_variance, kM_lengthscale, kM_likelihood_variance
        )
    except BaseException as ex:
        #如果超参数则不需要再次优化
        #若已经默认初始化
        if kM_variance == kM_lengthscale == kM_likelihood_variance ==1.0:
            raise BaseException(
                'Retry with default hyperparameters - already using default parameters.'
            ) from ex
        (
            kM_nlml,
            kM_params
        ) = fit_matern_kernel(time_series_data)

    is_cp_location_default = (
        (not kC_changepoint_location)
        or kC_changepoint_location < time_series_data['X'].iloc[0]
        or kC_changepoint_location > time_series_data['X'].iloc[-1]
    )
    if is_cp_location_default:
        # 初始化为区间中点
        kC_changepoint_location = (
            time_series_data['X'].iloc[-1] + time_series_data['X'].iloc[0]
        ) / 2.0
    
    if not k1_variance:
        k1_variance = kM_params['kM_variance']

    if not k1_lengthscale:
        k1_lengthscale = kM_params['kM_lengthscales']

    if not k2_variance:
        k2_variance = kM_params['kM_variance']

    if not k2_lengthscale:
        k2_lengthscale = kM_params['kM_lengthscales']

    if not kC_likelihood_variance:
        kC_likelihood_variance = kM_params['kM_likelihood_variance']

    try:
        (changepoint_location, kC_nlml, kC_params) = fit_changepoint_kernel(
            time_series_data,
            k1_variance=k1_variance,
            k1_lengthscale=k1_lengthscale,
            k2_variance=k2_variance,
            k2_lengthscale=k2_lengthscale,
            kC_likelihood_variance=kC_likelihood_variance,
            kC_changepoint_location=kC_changepoint_location,
            kC_steepness=kC_steepness,
        )
    except BaseException as ex:
        if(
            k1_variance
            == k1_lengthscale
            == k2_variance
            == k2_lengthscale
            == kC_likelihood_variance
            == kC_steepness
            == 1.0
        ) and is_cp_location_default:
            raise BaseException(
                'Retry with default hyperparameters - already using default parameters'
            ) from ex
        (
            changepoint_location,
            kC_nlml,
            kC_params,
        ) = fit_changepoint_kernel(time_series_data)

    cp_score = changepoint_severity(kC_nlml, kM_nlml)
    cp_loc_normalised = (time_series_data['X'].iloc[-1] - changepoint_location) / (
        time_series_data['X'].iloc[-1] - time_series_data['X'].iloc[0]
    )

    return cp_score, changepoint_location, cp_loc_normalised, kM_params, kC_params

def run_module(
    time_series_data: pd.DataFrame,
    lookback_window_length: int,
    output_csv_file_path: str,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    use_kM_hyp_to_initialise_kC = True,
):
    """运行拐点检测模块，结果输出为.csv文件

    参数:
        time_series_data (pd.DataFrame): 时间为index，日收益率为列的时间序列数据
        lookback_window_length (int): 回测窗口期
        output_csv_file_path (str): CSV扩展输出结果
        start_date (dt.datetime, optional): 模块的开始时间，初始为None。
        end_date (dt.datetime, optional): 模块的结束时间，初始为None。
        use_kM_hyp_to_initialise_kC (bool, optional): 使用拟合matn3 /2内核的参数初始化Changepoint内核参数。默认为True。
    """
    if start_date and end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1):,:
        ]
        remaining_data = time_series_data.loc[start_date:end_date,:]
        if remaining_data.empty:
            print(f"⚠️ 警告：数据在 {start_date} 之前已耗尽或为空，跳过此日期。")
            # 根据逻辑，这里可能需要 break, return 或者 continue
            # 假设这是在一个循环里，或者直接返回
            return  # 或者 break
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:,:]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()
    elif not start_date and not end_date:
        time_series_data = time_series_data.copy()
    elif not start_date:
        time_series_data = time_series_data.iloc[:end_date,:].copy()
    elif not end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1) :, :
        ]
        remaining_data = time_series_data.loc[start_date:, :]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:,:]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()

    # 获取 output_csv_file_path 的父目录（即 data/quandl_cpd_21lbw）
    output_dir = os.path.dirname(output_csv_file_path)

    # 如果这个目录不存在，就递归创建它
    os.makedirs(output_dir, exist_ok=True)

    csv_fields = ['date','t','cp_location','cp_location_norm','cp_score']
    with open(output_csv_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)

    time_series_data['date'] = time_series_data.index
    time_series_data = time_series_data.reset_index(drop=True)
    for window_end in range(lookback_window_length + 1, len(time_series_data)):
        ts_data_window = time_series_data.iloc[
            window_end - (lookback_window_length + 1) : window_end
        ][['date','daily_returns']].copy()
        ts_data_window['X'] = ts_data_window.index.astype(float)
        ts_data_window = ts_data_window.rename(columns={'daily_returns': 'Y'})
        time_index = window_end -1
        window_date = ts_data_window['date'].iloc[-1].strftime('%Y-%m-%d')

        try:
            if use_kM_hyp_to_initialise_kC:
                cp_score, cp_loc, cp_loc_normalised, _, _ =changepoint_loc_and_score(
                    ts_data_window,
                )
            else:
                cp_score, cp_loc, cp_loc_normalised, _, _ =changepoint_loc_and_score(
                    ts_data_window,
                    k1_lengthscale=1.0,
                    k1_variance=1.0,
                    k2_lengthscale=1.0,
                    k2_variance=1.0,
                    kC_likelihood_variance=1.0,
                )
        
        except Exception as e:
            # 打印具体的错误信息，方便调试
            print(f"❌ 计算失败 (Window End: {window_end}): {e}")
            import traceback
            traceback.print_exc() # 打印完整堆栈:
            # 失败则写NA
            cp_score, cp_loc, cp_loc_normalised = 'NA','NA','NA'

        with open(output_csv_file_path,'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [window_date, time_index,cp_loc, cp_loc_normalised, cp_score]
            )
