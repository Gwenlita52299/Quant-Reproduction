import torch
import warnings
# 忽略特定的 experimental warning
warnings.filterwarnings("ignore", message=".*considered *experimental*.*")
# 或者暴力屏蔽所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import os
import logging

# 1. 屏蔽 TensorFlow 的 C++底层日志 (只显示 Error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. 屏蔽 Python层面的 TensorFlow 警告
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# 3. 屏蔽 absl 库的警告 (TensorFlow 内部用了这个库)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pandas as pd
import datetime as dt

import MOM_TRANS.changepoint_detection as cpd
from MOM_TRANS.data_prep import cal_returns

from setting.default import CPD_DEFAULT_LBW, USE_KM_HYP_TO_INITIALISE_KC
import argparse
import gc



def main(
    ticker: str, output_file_path: str, start_date: dt.datetime, end_date: dt.datetime, lookback_window_length :int
):
    data = pd.read_csv(f"C:/Users/Administrator/Desktop/repo/Process/data/split/{ticker}.csv") 
    data.set_index('date', inplace=True)
    data.index = pd.to_datetime(data.index, errors='coerce')
    data["daily_returns"] = cal_returns(data["close"])
    data.sort_index(inplace=True)

    cpd.run_module(
        data, lookback_window_length, output_file_path, start_date, end_date, USE_KM_HYP_TO_INITIALISE_KC
    )
    del data 
    gc.collect() 


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "ticker",
            metavar="t",
            type=str,
            nargs="?",
            default="AP",
            # choices=[],
            help="Ticker type",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            nargs="?",
            default="data/test.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "start_date",
            metavar="s",
            type=str,
            nargs="?",
            default="2017-01-01",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "end_date",
            metavar="e",
            type=str,
            nargs="?",
            default="2025-12-31",
            help="End date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )

        args = parser.parse_known_args()[0]

        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")

        return (
            args.ticker,
            args.output_file_path,
            start_date,
            end_date,
            args.lookback_window_length
        )

    main(*get_args())
