import multiprocessing
import argparse
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import os
import logging
from tqdm import tqdm 

# 1. 屏蔽 TensorFlow 的 C++底层日志 (只显示 Error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. 屏蔽 Python层面的 TensorFlow 警告
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# 3. 屏蔽 absl 库的警告 (TensorFlow 内部用了这个库)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from setting.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = 2


def main(lookback_window_length: int):
    if not os.path.exists(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m Process.cpd_single "{ticker}" "{os.path.join(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "2017-01-01" "2025-12-31" "{lookback_window_length}"'
        for ticker in QUANDL_TICKERS
    ]
    with multiprocessing.Pool(processes=N_WORKERS) as process_pool:
        # 使用 imap_unordered 可以稍微提高点效率，因为不需要等待前一个任务按顺序完成
        # tqdm(iterable, total=总任务数, desc="描述文字", unit="单位")
        results = list(tqdm(
            process_pool.imap_unordered(os.system, all_processes), 
            total=len(all_processes), 
            desc="Processing Tickers", 
            unit="ticker"
        ))

    print("✅ All processes completed.")
    #process_pool = multiprocessing.Pool(processes=N_WORKERS)
    #process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
