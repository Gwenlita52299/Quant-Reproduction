import tensorflow as tf
from tensorflow import keras
import gc
import numpy as np

concat = keras.backend.concatenate
stack = keras.backend.stack
K = keras.backend
Add = keras.layers.Add
LayerNorm = keras.layers.LayerNormalization
Dense = keras.layers.Dense
Multiply = keras.layers.Multiply
Dropout = keras.layers.Dropout
Activation = keras.layers.Activation
Lambda = keras.layers.Lambda

from MOM_TRANS.deep_momentum_network import DeepMomentumNetworkModel, SharpeLoss
from setting.hyperparameter_grid import(
    HP_DROPOUT_RATE,
    HP_HIDDEN_LAYER_SIZE,
    HP_LEARNING_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_MINIBATCH_SIZE,
)

def tf_stack(x, axis=0):
    if not isinstance(x,list):
        x=[x]
    return K.stack(x,axis=axis)

def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    '''返回简单的Keras线性层

    参数:
        size: 输出大小
        activation: 激活功能
        use_time_distributed: 是否跨时间
        use_bias: 是否存在偏差
    ''' 
    linear = keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = keras.layers.TimeDistributed(linear)
    return linear

def apply_mlp(
    inputs,
    hidden_size,
    output_size,
    output_activation=None,
    hidden_activation='tanh',
    use_time_distributed=False,
):
    '''对输入应用简单前馈网络。

    参数:
        输入：MLP输入
        hidden_size：隐藏状态大小
        output_size: MLP的输出大小
        output_activation：应用于输出的激活函数
        hidden_activation：用于输入的激活函数
        use_time_distributed：是否跨时间应用
    返回:
        用于MLP输出的张量。
    ''' 
    if use_time_distributed:
        hidden = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_size, activation=hidden_activation)
        )(inputs)
        return keras.layers.Dense(
            keras.layers.Dense(output_size, activation=output_size)
        )(hidden)
    else:
        hidden = keras.layers.Dense(hidden_size,activation=hidden_activation)(inputs)
        return keras.layers.Dense(output_size,activation=output_activation)(hidden)

def apply_gating_layer(
    x,
    hidden_layer_size: int,
    dropout_rate: float = None,
    use_time_distributed: bool = True,
    activation = None,
):
    '''对输入应用门控线性单元

    参数:
        x: 输入门控层
        hidden_layer_size: GLU的维度
        dropout_rate: Dropout rate 如果有的话
        use_time_distributed: 是否应用于跨时间
        activation: 必要时应用于线性特征变换的激活函数
    输出:
        Tuple of tensors for: (GLU output, gate)
    '''

    if dropout_rate is not None:
        x = keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_layer_size, activation=activation)
        )(x)
        gated_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_layer_size, activation='sigmoid')
        )(x)
    else:
        activation_layer = keras.layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = keras.layers.Dense(hidden_layer_size, activation='sigmoid')(x)

    return keras.layers.multiply([activation_layer, gated_layer]), gated_layer

def add_and_norm(x_list):
    temp = keras.layers.Add()(x_list)
    temp = keras.layers.LayerNormalization()(temp)
    return temp

def gated_residual_network(
    x,
    hidden_layer_size: int,
    output_size: int = None,
    dropout_rate: float = None,
    use_time_distributed: bool = True,
    additional_context = None,
    return_gate: bool = False,
):
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = keras.layers.Dense(output_size)
        if use_time_distributed:
            linear = keras.layers.TimeDistributed(linear)
        skip = linear(x)

    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False,
        )(additional_context)
    hidden = keras.layers.Activation("elu")(hidden)
    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None,
    )

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])

def get_decoder_mask(self_attn_inputs):
    len_s = tf.shape(self_attn_inputs)[-2]
    bs = tf.shape(self_attn_inputs)[:-2]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), -2)
    return mask

class ScaledDotProductAttention(keras.layers.Layer):
    def __init__(self, attn_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = keras.layers.Dropout(attn_dropout)
        self.activation = keras.layers.Activation("softmax")

    def __call__(self, q, k, v, mask):
        attn = Lambda(tempering_batchdot)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e9) * (1.0 - tf.cast(x,'float32')))(
                mask
            )
            attn = keras.layers.add([attn, mmask])
        attn = self. activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn,v])
        return output, attn

def tempering_batchdot(input_list):
    d, k = input_list
    temper = tf.sqrt(tf.cast(k.shape[-1],dtype='float32'))
    return K.batch_dot(d,k,axes=[2,2]) / temper

class InterpretableMultiHeadAttention(keras.layers.Layer):
    """定义可解释多头注意力机制层

    参数:
        n_head: 头数
        d_k: Key/query 在头中的维度
        d_v: Value 的维度
        dropout: Dropout概率
        qs_layers: query的投影层
        ks_layers: key的投影层
        vs_layers: value的投影层
        attention: 缩放点积注意力层
        w_o: 输出层
    """
    def __init__(self, n_head: int, d_model: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        vs_layer = keras.layers.Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(keras.layers.Dense(d_k, use_bias=False))
            self.ks_layers.append(keras.layers.Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)

        self.attention = ScaledDotProductAttention()
        self.w_o = keras.layers.Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """应用可解释多头注意力机制
        用T在注意力机制中引入时间步
        
        参数:
            q: Query 张量形状为=(?, T, d_model)
            k: Key形状为=(?, T, d_model)
            v: Values形状为=(?, T, d_model)
            mask: 掩码如果需要的话，形状为(?, T, T)
        输出:
            (layer outputs, attention weights)的元组
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = keras.layers.Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = Lambda(tf_stack)(heads) if n_head > 1 else heads[0]
        attn = Lambda(tf_stack)(attns)

        outputs = Lambda(K.mean, arguments={'axis':0})(head) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = keras.layers.Dropout(self.dropout)(outputs)

        return outputs, attn

class TftDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE, **params):
        params = params.copy()

        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        # self._input_obs_loc = params['input_obs_loc']
        self._static_input_loc = params['static_input_loc']
        self._known_regulat_input_idx = params['known_regular_inputs']
        self._known_categorical_input_idx = params['known_categorical_inputs']
        self.category_counts = params['category_counts']

        self.column_definition = params['column_definition']
        
        self.num_encoder_steps = params['num_encoder_steps']
        self.num_stacks = params['stack_size']
        self.num_heads = params['num_heads']
        self.input_size = int(params['input_size'])

        super().__init__(project_name, hp_directory, hp_minibatch_size, ** params)

    def model_builder(self, hp):
        self.hidden_layer_size = hp.Choice(
            'hidden_layer_size', values=HP_HIDDEN_LAYER_SIZE
        )
        self.dropout_rate = hp.Choice('dropout_rate', values=HP_DROPOUT_RATE)
        self.max_gradient_norm = hp.Choice(
            'max_gradient_norm', values=HP_MAX_GRADIENT_NORM
        )
        self.learning_rate = hp.Choice('learning_rate', values=HP_LEARNING_RATE)

        time_steps = self.time_steps
        combined_input_size = self.input_size
        # encoder_steps = self.num_encoder_steps

        all_inputs = keras.layers.Input(
            shape=(
                time_steps,
                combined_input_size,
            ),
            name='input',
        )

        (
            unknown_inputs,
            known_combined_layer,
            # obs_inputs,
            static_inputs,
        ) = self.get_tft_embeddings(all_inputs)

        if unknown_inputs is not None:
            historical_inputs = concat(
                [
                    unknown_inputs,
                    known_combined_layer,
                ],
                axis=-1,
            )
        else:
            historical_inputs = concat(
                [
                    known_combined_layer,
                ],
                axis=-1
            )

        def static_combine_and_mask(embedding):
            """对静态输入应用选择网络

            参数:
                embedding: 转换后的静态输入

            输出:
                变量选择网络的张量输出
            """
            _, num_static, static_dim = embedding.get_shape().as_list()[-3:]

            shape = tf.shape(embedding)
            flatten = tf.reshape(
                embedding, tf.concat([shape[:-2],[num_static * static_dim]], axis=-1)
            )

            #门控残差网络的非线性变换
            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None,
            )

            sparse_weights = keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = Lambda(tf.expand_dims, arguments={'axis':-1})(
                sparse_weights
            )

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(
                    embedding[:,i:i+1,:],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                )
                trans_emb_list.append(e)

            transformed_embedding = (
                keras.layers.Concatenate(axis=1)(trans_emb_list)
                if len(trans_emb_list) > 1
                else trans_emb_list[0]
            )

            combined = keras.layers.multiply([sparse_weights, transformed_embedding])
            
            static_vec = Lambda(K.sum, arguments={'axis': 1})(combined)

            return static_vec, sparse_weights

        static_encoder, static_weights = static_combine_and_mask(static_inputs)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False
        )

        def LSTM_combine_and_mask(embedding):
            time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()[-3:]

            batch_dimensions = tf.shape(embedding)[:-3]
            new_shape = tf.concat(
                [batch_dimensions, [time_steps, embedding_dim * num_inputs]], axis=-1
            )
            flatten = tf.reshape(embedding, shape=new_shape)

            if static_inputs is not None:
                expanded_static_context = Lambda(tf.expand_dims, arguments={'axis':1})(
                    static_context_variable_selection
                )
            else:
                expanded_static_context = None

            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True,
            )

            sparse_weights = keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = Lambda(tf.expand_dims, arguments={'axis':-2})(
                sparse_weights
            )

            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[...,i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                )
                trans_emb_list.append(grn_output)

            transformed_embedding = Lambda(tf_stack, arguments={'axis':-1})(
                trans_emb_list
            )

            combined = keras.layers.multiply([sparse_weights, transformed_embedding])
            temporal_ctx = Lambda(K.sum, arguments={'axis':-1})(combined)

            return temporal_ctx, sparse_weights, static_gate

        input_embeddings, flags, _ = LSTM_combine_and_mask(historical_inputs)

        def get_lstm(return_state):
            lstm = keras.layers.LSTM(
                self.hidden_layer_size,
                return_sequences=True,
                return_state=return_state,
                stateful=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
            )
            return lstm

        lstm_layer = get_lstm(return_state=False)(
            input_embeddings,
            initial_state=[static_context_state_h, static_context_state_c]
        )

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None
        )
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        expanded_static_context = Lambda(tf.expand_dims, arguments={'axis': -2})(
            static_context_enrichment
        )

        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True,
        )

        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate
        )

        mask = get_decoder_mask(enriched)
        x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)

        x, _ = apply_gating_layer(
            x, self.hidden_layer_size, dropout_rate=self.dropout_rate, activation=None
        )
        x = add_and_norm([x, enriched])

        decoder = gated_residual_network(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )

        decoder, _ = apply_gating_layer(
            decoder, self.hidden_layer_size, activation=None
        )
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        attention_components = {
            'decoder_self_attn': self_att,
            'static_flags': static_weights[...,0] if static_inputs is not None else [],
            'historical_flags': flags[...,0,:],
            'future_flags': flags[...,0,:],
        }

        if self.force_output_sharpe_length:
            outputs = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    self.output_size,
                    activation=tf.nn.tanh,
                    kernel_constraint=keras.constraints.max_norm(3),
                )
            )(transformer_layer[Ellipsis, -self.force_output_sharpe_length:, :])
        else:
            outputs = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    self.output_size,
                    activation=tf.nn.tanh,
                    kernel_constraint=keras.constraints.max_norm(3),
                )
            )(transformer_layer[Ellipsis, :, :])

        self._attention_components = attention_components

        adam = keras.optimizers.Adam(
            lr=self.learning_rate, clipnorm=self.max_gradient_norm
        )

        model = keras.Model(inputs=all_inputs, outputs=outputs)

        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(loss=sharpe_loss, optimizer=adam, sample_weight_mode='temporal')

        self._input_placeholder = all_inputs

        return model
    
    def get_tft_embeddings(self, all_inputs):
        time_steps = self.time_steps
        
        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    all_inputs.get_shape().as_list()[-1], self.input_size
                )
            )

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_size = [
            self.hidden_layer_size for _, _ in enumerate(self.category_counts)
        ]

        embeddings = []
        for i in range(num_categorical_variables):

            embedding = keras.Sequential(
                [
                    keras.layers.InputLayer([time_steps]),
                    keras.layers.Embedding(
                        self.category_counts[i],
                        embedding_size[i],
                        input_length=time_steps,
                        dtype=tf.float32,
                    ),
                ]
            )
            embeddings.append(embedding)

        regular_inputs, categorical_inputs = (
            all_inputs[:, :, :num_regular_variables],
            all_inputs[:,:, num_regular_variables:],
        )

        embedded_inputs = [
            embeddings[i](categorical_inputs[Ellipsis, i])
            for i in range(num_categorical_variables)
        ]

        if self._static_input_loc:
            static_inputs = [
                keras.layers.Dense(self.hidden_layer_size)(
                    regular_inputs[:, 0, i : i + 1]
                )
                for i in range(num_regular_variables)
                if i in self._static_input_loc
            ] + [
                embedded_inputs[i][:, 0, :]
                for i in range(num_categorical_variables)
                if i + num_regular_variables in self._static_input_loc
            ]
            static_inputs = keras.backend.stack(static_inputs, axis=1)
        
        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            return keras.layers.TimeDistributed(
                keras.layers.Dense(self.hidden_layer_size)
            )(x)
        
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if(
                i not in self._known_categorical_input_idx
                # and i + num_regular_variables not in self._inputs_obs_loc
            ):
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if(
                i not in self._known_regulat_input_idx
                # and i not in self._input_obs_loc
            ):
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = keras.backed.stack(
                unknown_inputs + wired_embeddings, axis=-1
            )
        else:
            unknown_inputs = None

        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1])
            for i in self._known_regulat_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = keras.backend.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1
        )

        return unknown_inputs, known_combined_layer, static_inputs
    
    def get_attention(self, data, batch_size, mask = None):
        if mask:
            inputs = data['inputs'][mask]
            identifiers = data['identifier'][mask]
            time = data['date'][mask]
        else:
            inputs = data['inputs']
            identifiers = data['identifier']
            time = data['date']

        def get_batch_attention_weights(input_batch):
            input_placeholder = self._input_placeholder
            attention_weights = {}

            for k in self._attention_components:
                extractor = tf.keras.Model(
                    inputs=input_placeholder, outputs=self._attention_components[k]
                )
                attention_weight = extractor(input_batch.astype(np.float32))
                attention_weights[k] = attention_weight
            return attention_weights
        
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches +=1

        batched_inputs = [
            inputs[i * batch_size : (i+1) * batch_size, Ellipsis]
            for i in range(num_batches)
        ]

        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in batched_inputs
        ]
        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])

            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:, 0, 0]
        attention_weights['time'] = time[:,:,0]

        return attention_weights