import os
import json
import pathlib
import shutil
import copy

from keras_tuner.tuners.randomsearch import RandomSearch
from abc import ABC, abstractmethod

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import collections

import keras_tuner as kt

from setting.hyperparameter_grid import (
    HP_HIDDEN_LAYER_SIZE,
    HP_DROPOUT_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_LEARNING_RATE,
    HP_MINIBATCH_SIZE,
)

from setting.fixed_parameter import MODLE_PARAMS

from MOM_TRANS.model_input import ModelFeatures
from empyrical import sharpe_ratio

def validate_features(data, labels, active_flags, identifier, time, stage="未知"):
    """
    完整的数据验证函数
    stage: "训练集" 或 "验证集"
    """
    print(f"\n{'='*60}")
    print(f"[数据检查] {stage}")
    print(f"{'='*60}")
    
    # 1. 检查 NaN
    data_nan_count = np.isnan(data).sum()
    labels_nan_count = np.isnan(labels).sum()
    flags_nan_count = np.isnan(active_flags).sum()
    
    print(f"✓ NaN 检查:")
    print(f"  - data 中的 NaN 数量: {data_nan_count}")
    print(f"  - labels 中的 NaN 数量: {labels_nan_count}")
    print(f"  - active_flags 中的 NaN 数量: {flags_nan_count}")
    
    if data_nan_count > 0 or labels_nan_count > 0:
        raise ValueError(f"❌ {stage} 包含 NaN 值！")
    
    # 2. 检查 Inf
    data_inf_count = np.isinf(data).sum()
    labels_inf_count = np.isinf(labels).sum()
    
    print(f"✓ Inf 检查:")
    print(f"  - data 中的 Inf 数量: {data_inf_count}")
    print(f"  - labels 中的 Inf 数量: {labels_inf_count}")
    
    if data_inf_count > 0 or labels_inf_count > 0:
        raise ValueError(f"❌ {stage} 包含 Inf 值！")
    
    # 3. 数值范围检查
    print(f"✓ 数值范围:")
    print(f"  - data: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  - labels: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"  - active_flags: [{active_flags.min():.4f}, {active_flags.max():.4f}]")
    
    # 警告：如果数值范围异常大
    if abs(data.max()) > 1e6 or abs(data.min()) > 1e6:
        print(f"  ⚠️  警告: data 数值范围过大，可能需要归一化")
    
    # 4. 方差检查（零方差特征无用）
    data_reshaped = data.reshape(-1, data.shape[-1])
    feature_std = np.std(data_reshaped, axis=0)
    zero_var_features = np.where(feature_std < 1e-8)[0]
    
    print(f"✓ 方差检查:")
    print(f"  - 零方差特征数量: {len(zero_var_features)}")
    if len(zero_var_features) > 0:
        print(f"  ⚠️  特征索引 {zero_var_features} 方差接近 0")
    
    # 5. 标签分布检查
    print(f"✓ 标签统计:")
    print(f"  - 均值: {labels.mean():.6f}")
    print(f"  - 标准差: {labels.std():.6f}")
    print(f"  - 最大值: {labels.max():.6f}")
    print(f"  - 最小值: {labels.min():.6f}")
    
    # 6. Active flags 检查
    active_ratio = active_flags.mean()
    print(f"✓ Active Flags:")
    print(f"  - 激活比例: {active_ratio:.2%}")
    if active_ratio < 0.1:
        print(f"  ⚠️  警告: 激活样本过少 (<10%)")
    
    # 7. 形状检查
    print(f"✓ 数据形状:")
    print(f"  - data: {data.shape}")
    print(f"  - labels: {labels.shape}")
    print(f"  - identifier: {identifier.shape}")
    print(f"  - time: {time.shape}")
    
    print(f"{'='*60}\n")


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        super().__init__()
    '''
    def call(self, y_true, weights):
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        return -(
            mean_returns
            / tf.sqrt(
                tf.reduce_mean(tf.square(captured_returns))
                - tf.square(mean_returns)
                + 1e-9
            )
            * tf.sqrt(252.0)
        )
    '''
    def call(self, y_true, weights):
        # 数值检查（保留你之前添加的）
        tf.debugging.check_numerics(weights, "weights 包含 NaN/Inf")
        tf.debugging.check_numerics(y_true, "y_true 包含 NaN/Inf")
        
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        
        # 使用稳定的方差计算
        variance = tf.math.reduce_variance(captured_returns)
        
        # 增大 epsilon 防止除零
        epsilon = 1e-6  # 从 1e-9 增大到 1e-6
        sharpe = mean_returns / tf.sqrt(variance + epsilon) * tf.sqrt(252.0)
        
        # 裁剪 Sharpe 值防止极端梯度
        sharpe = tf.clip_by_value(sharpe, -10.0, 10.0)
        
        return -sharpe


class SharpeValidationLoss(keras.callbacks.Callback):
    # TODO check if weights already exist and pass in best sharpe
    def __init__(
        self,
        inputs,
        returns,
        time_indices,
        num_time,  # including a count for nulls which will be indexed as 0
        early_stopping_patience,
        n_multiprocessing_workers,
        weights_save_location="tmp/checkpoint",
        # verbose=0,
        min_delta=1e-4,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = np.NINF  # since calculating positive Sharpe...
        # self.best_weights = None
        self.weights_save_location = weights_save_location
        # self.verbose = verbose

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        import os
        import pathlib
        positions = self.model.predict(
                self.inputs,
                workers=self.n_multiprocessing_workers,
                use_multiprocessing=True,
            )

        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]

        sharpe = (
            tf.reduce_mean(captured_returns)
            / tf.sqrt(
                tf.math.reduce_variance(captured_returns)
                + tf.constant(1e-9, dtype=tf.float64)
            )
            * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0
            
            # 【修改】确保目录存在 + 处理编码
           
            
            # 转换为 pathlib.Path 处理编码问题
            save_path = pathlib.Path(self.weights_save_location)
            
            # 确保父目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用字符串路径保存（避免编码问题）
            try:
                self.model.save_weights(str(save_path))
                print(f"✓ 权重已保存到: {save_path}")
            except Exception as e:
                print(f"❌ 保存权重失败: {e}")
                # 尝试备用路径（临时目录）
                import tempfile
                backup_path = pathlib.Path(tempfile.gettempdir()) / f"model_weights_epoch_{epoch}.h5"
                self.model.save_weights(str(backup_path))
                self.weights_save_location = str(backup_path)
                print(f"⚠️  使用备用路径: {backup_path}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.load_weights(str(pathlib.Path(self.weights_save_location)))
        
        logs["sharpe"] = sharpe
        print(f"\nval_sharpe {logs['sharpe']:.4f}")


# Tuner = RandomSearch
class TunerValidationLoss(kt.tuners.RandomSearch):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)


class TunerDiversifiedSharpe(kt.tuners.RandomSearch):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        for callback in original_callbacks:
            if isinstance(callback, SharpeValidationLoss):
                callback.set_weights_save_loc(
                    self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                )

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            # callbacks.append(model_checkpoint)
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )


class DeepMomentumNetworkModel(ABC):
    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):
        params = params.copy()

        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])
        # self.sliding_window = params["sliding_window"]
        self.random_search_iterations = params["random_search_iterations"]
        self.evaluate_diversified_val_sharpe = params["evaluate_diversified_val_sharpe"]
        self.force_output_sharpe_length = params["force_output_sharpe_length"]

        print("Deep Momentum Network params:")
        for k in params:
            print(f"{k} = {params[k]}")

        # To build model
        def model_builder(hp):
            return self.model_builder(hp)

        if self.evaluate_diversified_val_sharpe:
            self.tuner = TunerDiversifiedSharpe(
                model_builder,
                # objective="val_loss",
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )
        else:
            self.tuner = TunerValidationLoss(
                model_builder,
                objective="val_loss",
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )

    @abstractmethod
    def model_builder(self, hp):
        return

    @staticmethod
    def _index_times(val_time):
        val_time_unique = np.sort(np.unique(val_time))
        if val_time_unique[0]:  # check if ""
            val_time_unique = np.insert(val_time_unique, 0, "")
        mapping = dict(zip(val_time_unique, range(len(val_time_unique))))

        @np.vectorize
        def get_indices(t):
            return mapping[t]

        return get_indices(val_time), len(mapping)
    



    def hyperparameter_search(self, train_data, valid_data):
        data, labels, active_flags, identifier, time = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, val_identifier, val_time = ModelFeatures._unpack(valid_data)

        validate_features(data, labels, active_flags, identifier, time, stage="训练集")
        validate_features(val_data, val_labels, val_flags, val_identifier, val_time, stage="验证集")

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                # batch_size=minibatch_size,
                # covered by Tuner class
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
            )
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                ),
                # tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                # batch_size=minibatch_size,
                # covered by Tuner class
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags,
                ),
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
                # validation_batch_size=1,
            )

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
        best_model = self.tuner.get_best_models(num_models=1)[0]
        return best_hp, best_model

    def load_model(
        self,
        hyperparameters,
    ) -> tf.keras.Model:
        hyp = kt.engine.hyperparameters.HyperParameters()
        hyp.values = hyperparameters
        return self.tuner.hypermodel.build(hyp)

    def fit(
        self,
        train_data: np.array,
        valid_data: np.array,
        hyperparameters,
        temp_folder: str,
    ):
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        model = self.load_model(hyperparameters)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                    weights_save_location=temp_folder,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
            )
            model.load_weights(temp_folder)
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags,
                ),
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
            )
        return model

    def evaluate(self, data, model):
        """Applies evaluation metric to the training data.

        Args:
          data: Dataframe for evaluation
          eval_metric: Evaluation metic to return, based on model definition.

        Returns:
          Computed evaluation loss.
        """

        inputs, outputs, active_entries, _, _ = ModelFeatures._unpack(data)

        if self.evaluate_diversified_val_sharpe:
            _, performance = self.get_positions(data, model, False)
            return performance

        else:
            metric_values = model.evaluate(
                x=inputs,
                y=outputs,
                sample_weight=active_entries,
                workers=32,
                use_multiprocessing=True,
            )

            metrics = pd.Series(metric_values, model.metrics_names)
            return metrics["loss"]

    def get_positions(
        self,
        data,
        model,
        sliding_window=True,
        years_geq=np.iinfo(np.int32).min,
        years_lt=np.iinfo(np.int32).max,
    ):
        inputs, outputs, _, identifier, time = ModelFeatures._unpack(data)
        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            )  # TODO to_datetime maybe not needed
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        mask = (years >= years_geq) & (years < years_lt)

        positions = model.predict(
            inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True,  # , batch_size=1
        )
        if sliding_window:
            positions = positions[:, -1, 0].flatten()
        else:
            positions = positions.flatten()

        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "identifier": identifier[mask],
                "time": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask],
            }
        )

        # don't need to divide sum by n because not storing here
        # mean does not work as well (related to days where no information)
        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())

        return results, performance


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
        self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE, **params
    ):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE)
        # minibatch_size = hp.Choice("hidden_layer_size", HP_MINIBATCH_SIZE)

        input = keras.Input((self.time_steps, self.input_size))
        lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )(input)
        dropout = keras.layers.Dropout(dropout_rate)(lstm)

        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )(dropout[..., :, :])

        model = keras.Model(inputs=input, outputs=output)

        adam = keras.optimizers.Adam(
            learning_rate=learning_rate,  # 使用 learning_rate 而不是 lr（避免弃用警告）
            clipvalue=0.5,  # 改用 clipvalue，设置较小的值
        )

        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        return model
