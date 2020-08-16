"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""

import functools

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention,\
	BahdanauMonotonicAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, random_ops, variable_scope


#From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer, prev_max_attentions):
	"""Computes the attention and alignments for a given attention_mechanism."""

	if isinstance(attention_mechanism, LocationSensitiveAttention):
		alignments, next_attention_state, max_attentions = attention_mechanism(
			cell_output, state=attention_state, prev_max_attentions=prev_max_attentions)
	elif isinstance(attention_mechanism, BahdanauStepwiseMonotonicAttention):
		alignments, next_attention_state = attention_mechanism(
			cell_output, state=attention_state)
		max_attentions = tf.argmax(alignments, -1, output_type=tf.int32) # (N, Ty/r)
	else:
		raise '{} not supported!'.format(type(attention_mechanism))

	# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# Context is the inner product of alignments and values along the
	# memory time dimension.
	# alignments shape is
	#   [batch_size, 1, memory_time]
	# attention_mechanism.values shape is
	#   [batch_size, memory_time, memory_size]
	# the batched matmul is over memory_time, so the output shape is
	#   [batch_size, 1, memory_size].
	# we then squeeze out the singleton dim.
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state, max_attentions


def _location_sensitive_score(W_query, W_fil, W_keys):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################

	Args:
		W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
		W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
		W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
	Returns:
		A '[batch_size, max_time]' attention score (energy)
	"""
	# Get the number of hidden units from the trailing dimension of keys
	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable(
		'attention_variable_projection', shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		'attention_bias', shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])

def _smoothing_normalization(e):
	"""Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	Usually referred to as "hybrid" attention (content-based + location-based)
	Extends the additive attention described in:
	"D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
	to use previous alignments as additional location features.

	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
	"""

	def __init__(self,
				 num_units,
				 memory,
				 hparams,
				 is_training,
				 mask_encoder=True,
				 memory_sequence_length=None,
				 smoothing=False,
				 cumulate_weights=True,
				 name='LocationSensitiveAttention'):
		"""Construct the Attention mechanism.
		Args:
			num_units: The depth of the query mechanism.
			memory: The memory to query; usually the output of an RNN encoder.  This
				tensor should be shaped `[batch_size, max_time, ...]`.
			mask_encoder (optional): Boolean, whether to mask encoder paddings.
			memory_sequence_length (optional): Sequence lengths for the batch entries
				in memory.  If provided, the memory tensor rows are masked with zeros
				for values past the respective sequence lengths. Only relevant if mask_encoder = True.
			smoothing (optional): Boolean. Determines which normalization function to use.
				Default normalization function (probablity_fn) is softmax. If smoothing is
				enabled, we replace softmax with:
						a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
				Introduced in:
					J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
				  gio, “Attention-based models for speech recognition,” in Ad-
				  vances in Neural Information Processing Systems, 2015, pp.
				  577–585.
				This is mainly used if the model wants to attend to multiple input parts
				at the same decoding step. We probably won't be using it since multiple sound
				frames may depend on the same character/phone, probably not the way around.
				Note:
					We still keep it implemented in case we want to test it. They used it in the
					paper in the context of speech recognition, where one phoneme may depend on
					multiple subsequent sound frames.
			name: Name to use when creating ops.
		"""
		#Create normalization function
		#Setting it to None defaults in using softmax
		normalization_function = _smoothing_normalization if (smoothing == True) else None
		memory_length = memory_sequence_length if (mask_encoder==True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=memory_length,
				probability_fn=normalization_function,
				name=name)

		self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
			# kernel_size=hparams.attention_kernel, padding='same', use_bias=True,
			kernel_size=hparams.attention_kernel, padding='same', use_bias=False,
			bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
			dtype=tf.float32, name='location_features_layer')
		self._cumulate = cumulate_weights
		self.synthesis_constraint = hparams.synthesis_constraint and not is_training
		self.attention_win_size = tf.convert_to_tensor(hparams.attention_win_size, dtype=tf.int32)
		self.constraint_type = hparams.synthesis_constraint_type

	def __call__(self, query, state, prev_max_attentions):
		"""Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory's `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)

			# processed_location_features shape [batch_size, max_time, attention dimension]
			# [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			# location features [batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			# Projected location features [batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)

			# energy shape [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

		if self.synthesis_constraint:
			Tx = tf.shape(energy)[-1]
			# prev_max_attentions = tf.squeeze(prev_max_attentions, [-1])
			if self.constraint_type == 'monotonic':
				key_masks = tf.sequence_mask(prev_max_attentions, Tx)
				reverse_masks = tf.sequence_mask(Tx - self.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
			else:
				assert self.constraint_type == 'window'
				key_masks = tf.sequence_mask(prev_max_attentions - (self.attention_win_size // 2 + (self.attention_win_size % 2 != 0)), Tx)
				reverse_masks = tf.sequence_mask(Tx - (self.attention_win_size // 2) - prev_max_attentions, Tx)[:, ::-1]
			
			masks = tf.logical_or(key_masks, reverse_masks)
			paddings = tf.ones_like(energy) * (-2 ** 32 + 1)  # (N, Ty/r, Tx)
			energy = tf.where(tf.equal(masks, False), energy, paddings)

		# alignments shape = energy shape = [batch_size, max_time]
		alignments = self._probability_fn(energy, previous_alignments)
		max_attentions = tf.argmax(alignments, -1, output_type=tf.int32) # (N, Ty/r)

		# Cumulate alignments
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments

		return alignments, next_state, max_attentions


# source : https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
# Implementation for https://arxiv.org/abs/1906.00672

# Tips: The code could be directly used in place of BadahnauMonotonicAttention in Tensorflow codes. Similar to its 
# base class in the Tensorflow seq2seq codebase,  you may use "hard" for hard inference, or "parallel" for training or 
# soft inference. "recurrent" mode in BadahnauMonotonicAttention is not supported. 

# If you have already trained another model using BadahnauMonotonicAttention, the model could be reused, otherwise you 
# possibly have to tune the score_bias_init, which, similar to that in Raffel et al., 2017, is determined a priori to 
# suit the moving speed of the alignments, i.e. speed of speech of your training corpus in TTS cases. So 
# score_bias_init=3.5, is a good one for our data, but not necessarily for yours, and our experiments find that the 
# results are sensitive to this bias: When the parameter is deviated from the best value, by, say, a small amount of 
# 0.5, the whole training process may fail. sigmoid_noise=2.0 is enough in our experiments, but if you found that the 
# resultant alignments are far from binary, adding more noise (or annealing the noise) might be useful. Other 
# hyperparameters in our experiments simply follow the original Tacotron2 settings, and they work. 

def monotonic_stepwise_attention(p_choose_i, previous_attention, mode):
    # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
    # p_choose_i: probability to keep attended to the last attended entry i
    if mode == "parallel":
        pad = tf.zeros([tf.shape(p_choose_i)[0], 1], dtype=p_choose_i.dtype)
        attention = previous_attention * p_choose_i + tf.concat(
            [pad, previous_attention[:, :-1] * (1.0 - p_choose_i[:, :-1])], axis=1)
    elif mode == "hard":
        # Given that previous_alignments is one_hot
        move_next_mask = tf.concat([tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]], axis=1)
        stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1) # [B]
        attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
    else:
        raise ValueError("mode must be 'parallel', or 'hard'.")
    return attention


def _stepwise_monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode, seed=None):
    if sigmoid_noise > 0:
        noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                         seed=seed)
        score += sigmoid_noise * noise
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = math_ops.cast(score > 0, score.dtype)
    else:
        p_choose_i = math_ops.sigmoid(score)
    alignments = monotonic_stepwise_attention(p_choose_i, previous_alignments, mode)
    return alignments


class BahdanauStepwiseMonotonicAttention(BahdanauMonotonicAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=True,
                 score_mask_value=None,
                 sigmoid_noise=2.0,
                 sigmoid_noise_seed=None,
                 score_bias_init=3.5,
                 mode="parallel",
                 dtype=None,
                 name="BahdanauStepwiseMonotonicAttention"):
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _stepwise_monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=tf.layers.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=tf.layers.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init
