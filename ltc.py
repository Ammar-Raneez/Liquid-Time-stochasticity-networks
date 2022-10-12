'''
A TensorFlow V2 implementation of the Liquid Time-Constant cell proposed
by Hasani et al., (2020) at https://arxiv.org/pdf/2006.04439.pdf

An RNN with with continuous-time hidden
states determined by ordinary differential equations
'''

import tensorflow as tf
import numpy as np
from enum import Enum

class MappingType(Enum):
	Identity = 0
	Linear = 1
	Affine = 2

class ODESolver(Enum):
	SemiImplicit = 0
	Explicit = 1
	RungeKutta = 2

class LTCCell(tf.keras.layers.Layer):

	def __init__(self, units, **kwargs):
		'''
		Initializes the LTC cell & parameters
		Calls parent Layer constructor to initialize required fields
		'''

		super(LTCCell, self).__init__(**kwargs)
		self.input_size = -1
		self.units = units
		self.built = False

		# Number of ODE solver steps in one RNN step
		self._ode_solver_unfolds = 6
		self._solver = ODESolver.SemiImplicit

		self._input_mapping = MappingType.Affine

		self._erev_init_factor = 1

		self._w_init_max = 1.0
		self._w_init_min = 0.01
		self._cm_init_min = 0.5
		self._cm_init_max = 0.5
		self._gleak_init_min = 1
		self._gleak_init_max = 1

		self._w_min_value = 0.00001
		self._w_max_value = 1000
		self._gleak_min_value = 0.00001
		self._gleak_max_value = 1000
		self._cm_t_min_value = 0.000001
		self._cm_t_max_value = 1000

		self._fix_cm = None
		self._fix_gleak = None
		self._fix_vleak = None

		self._input_weights = None
		self._input_biases = None

	@property
	def state_size(self):
		return self.units

	def build(self, input_shape):
		'''
		Automatically triggered the first time __call__ is run
		'''

		self.input_size = int(input_shape[-1])
		self._get_variables()
		self.built = True

	@tf.function
	def call(self, inputs, states):
		'''
		Automatically calls build() the first time.
		Runs the LTC cell for one step using the previous RNN cell output & state
		by calculating a type of ODE solver to generate the next output and state
		'''

		inputs = self._map_inputs(inputs)

		if self._solver == ODESolver.Explicit:
				next_state = self._ode_step_explicit(
					inputs,
					states,
					_ode_solver_unfolds = self._ode_solver_unfolds
				)
		elif self._solver == ODESolver.SemiImplicit:
				next_state = self._ode_step_hybrid_euler(inputs, states)
		elif self._solver == ODESolver.RungeKutta:
				next_state = self._ode_step_runge_kutta(inputs, states)
		else:
				raise ValueError(f'Unknown ODE solver \'{str(self._solver)}\'')

		output = next_state
		return output, next_state

	def get_config(self):
		'''
		Enable serialization
		'''

		config = super(LTCCell, self).get_config()
		config.update({ 'units': self.units })
		return config

	# Helper methods
	def _get_variables(self):
		'''
		Creates the variables to be used within __call__
		'''

		self.sensory_mu = tf.Variable(
			tf.random.uniform(
				[self.input_size, self.units],
				minval = 0.3,
				maxval = 0.8,
				dtype = tf.float32
			),
			name = 'sensory_mu',
			trainable = True,
		)

		self.sensory_sigma = tf.Variable(
			tf.random.uniform(
				[self.input_size, self.units],
				minval = 3.0,
				maxval = 8.0,
				dtype = tf.float32
			),
			name = 'sensory_sigma',
			trainable = True,
		)

		self.sensory_W = tf.Variable(
			tf.constant(
				np.random.uniform(
					low = self._w_init_min,
					high = self._w_init_max,
					size = [self.input_size, self.units]
				),
				dtype = tf.float32
			),
			name = 'sensory_W',
			trainable = True,
			shape = [self.input_size, self.units]
		)

		sensory_erev_init = 2 * np.random.randint(
			low = 0,
			high = 2,
			size = [self.input_size, self.units]
		) - 1
		self.sensory_erev = tf.Variable(
			tf.constant(
				sensory_erev_init * self._erev_init_factor,
				dtype = tf.float32
			),
			name = 'sensory_erev',
			trainable = True,
			shape = [self.input_size, self.units]
		)

		self.mu = tf.Variable(
			tf.random.uniform(
				[self.units, self.units],
				minval = 0.3,
				maxval = 0.8,
				dtype = tf.float32)
			,
			name = 'mu',
			trainable = True,
		)

		self.sigma = tf.Variable(
			tf.random.uniform(
				[self.units, self.units],
				minval = 3.0,
				maxval = 8.0,
				dtype = tf.float32
			),
			name = 'sigma',
			trainable = True,
		)

		self.W = tf.Variable(
			tf.constant(
				np.random.uniform(
					low = self._w_init_min,
					high = self._w_init_max,
					size = [self.units, self.units]
				),
				dtype = tf.float32
			),
			name = 'W',
			trainable = True,
			shape = [self.units, self.units]
		)

		erev_init = 2 * np.random.randint(
			low = 0,
			high = 2,
			size = [self.units, self.units]
		) - 1
		self.erev = tf.Variable(
			tf.constant(
				erev_init * self._erev_init_factor,
				dtype = tf.float32
			),
			name = 'erev',
			trainable = True,
			shape = [self.units, self.units]
		)

		if self._fix_vleak is None:
			self.vleak = tf.Variable(
				tf.random.uniform(
					[self.units],
					minval = -0.2,
					maxval = 0.2,
					dtype = tf.float32
				),
				name = 'vleak',
				trainable = True,
			)
		else:
			self.vleak = tf.Variable(
				tf.constant(self._fix_vleak, dtype = tf.float32),
				name = 'vleak',
				trainable = False,
				shape = [self.units]
			)

		if self._fix_gleak is None:
			initializer = tf.constant(self._gleak_init_min, dtype = tf.float32)

			if self._gleak_init_max > self._gleak_init_min:
				initializer = tf.random.uniform(
					[self.units],
					minval = self._gleak_init_min,
					maxval = self._gleak_init_max,
					dtype = tf.float32
				)

			self.gleak = tf.Variable(
				initializer,
				name = 'gleak',
				trainable = True,
			)
		else:
			self.gleak = tf.Variable(
				tf.constant(self._fix_gleak),
				name = 'gleak',
				trainable = False,
				shape = [self.units]
			)

		if self._fix_cm is None:
			initializer = tf.constant(self._cm_init_min, dtype = tf.float32)

			if self._cm_init_max > self._cm_init_min:
				initializer = tf.random.uniform(
					[self.units],
					minval = self._cm_init_min,
					maxval = self._cm_init_max,
					dtype = tf.float32
				)

			self.cm_t = tf.Variable(
				initializer,
				name = 'cm_t',
				trainable = True,
			)
		else:
			self.cm_t = tf.Variable(
				tf.constant(self._fix_cm),
				name = 'cm_t',
				trainable = False,
				shape = [self.units]
			)

	def _map_inputs(self, inputs):
		'''
		Maps the inputs to the sensory layer
		Initializes weights & biases to be used
		'''

		# Create a workaround from creating tf Variables every function call
		# init with None and set only if not None - aka only first time
		if self._input_weights is None:
			self._input_weights = tf.Variable(
				# cannot directly assign variables within a tf.function (call()), but possible through callbacks
				lambda: tf.ones(
					[self.input_size],
					dtype = tf.float32
				),
				name = 'input_weights',
				trainable = True
			)

		if self._input_biases is None:
			self._input_biases = tf.Variable(
				lambda: tf.zeros(
					[self.input_size],
					dtype = tf.float32
				),
				name = 'input_biases',
				trainable = True
			)

		if self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear:
			inputs = inputs * self._input_weights
		if self._input_mapping == MappingType.Affine:
			inputs = inputs + self._input_biases

		return inputs

	def _ode_step_hybrid_euler(self, inputs, states):
		'''
		Implement Euler ODE solver - first-order numerical procedure
		'''

		# State returned as -> tuple(Tensor)
		v_pre = states[0]

		sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
		sensory_rev_activation = sensory_w_activation * self.sensory_erev

		w_numerator_sensory = tf.reduce_sum(input_tensor = sensory_rev_activation, axis = 1)
		w_denominator_sensory = tf.reduce_sum(input_tensor = sensory_w_activation, axis = 1)

		for _ in range(self._ode_solver_unfolds):
			w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

			rev_activation = w_activation * self.erev

			w_numerator = tf.reduce_sum(input_tensor = rev_activation, axis = 1) + w_numerator_sensory
			w_denominator = tf.reduce_sum(input_tensor = w_activation, axis = 1) + w_denominator_sensory

			numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
			denominator = self.cm_t + self.gleak + w_denominator

			v_pre = numerator / denominator

		return v_pre

	def _ode_step_runge_kutta(self, inputs, states):
		'''
		Implement Runge-Kutta ODE solver - RK4, fourth-order numerical procedure
		'''

		h = 0.1
		for _ in range(self._ode_solver_unfolds):
			k1 = h * self._f_prime(inputs, states)
			k2 = h * self._f_prime(inputs, states + k1 * 0.5)
			k3 = h * self._f_prime(inputs, states + k2 * 0.5)
			k4 = h * self._f_prime(inputs, states + k3)

			states = states + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

		return states

	def _ode_step_explicit(self, inputs, states, _ode_solver_unfolds):
		'''
		Implement ODE explicit iterative solver - a generalization of RK4
		'''

		v_pre = states[0]

		# Pre-compute the effects of the sensory neurons
		sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
		w_reduced_sensory = tf.reduce_sum(input_tensor = sensory_w_activation, axis = 1)

		# Unfold the ODE multiple times into one RNN step
		for _ in range(_ode_solver_unfolds):
			f_prime = self._calculate_f_prime(v_pre, sensory_w_activation, w_reduced_sensory)
			v_pre = v_pre + 0.1 * f_prime

		return v_pre

	def _f_prime(self, inputs, states):
		'''
		Obtain f' for the ODE solvers
		'''

		v_pre = states[0]

		# Pre-compute the effects of the sensory neurons
		sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
		w_reduced_sensory = tf.reduce_sum(input_tensor = sensory_w_activation, axis = 1)
		f_prime = self._calculate_f_prime(v_pre, sensory_w_activation, w_reduced_sensory)

		return f_prime

	def _calculate_f_prime(self, v_pre, sensory_w_activation, w_reduced_sensory):
		'''
		Helper function to calculate f'
		'''

		# Unfold the ODE multiple times into one RNN step
		w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
		w_reduced_synapse = tf.reduce_sum(input_tensor = w_activation, axis = 1)
		sensory_in = self.sensory_erev * sensory_w_activation
		synapse_in = self.erev * w_activation
		sum_in = (
			tf.reduce_sum(input_tensor = sensory_in, axis = 1) -
				v_pre * w_reduced_synapse + tf.reduce_sum(input_tensor = synapse_in, axis = 1) - 
					v_pre * w_reduced_sensory
		)

		return 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

	def _sigmoid(self, v_pre, mu, sigma):
		v_pre = tf.reshape(v_pre, [-1, v_pre.shape[-1], 1])
		mues = v_pre - mu
		x = sigma * mues
		return tf.nn.sigmoid(x)

# References
# https://splunktool.com/how-can-i-implement-a-custom-rnn-specifically-an-esn-in-tensorflow
# https://colab.research.google.com/github/luckykadam/adder/blob/master/rnn_full_adder.ipynb
# https://www.tutorialexample.com/build-custom-rnn-by-inheriting-rnncell-in-tensorflow-tensorflow-tutorial/
# https://notebook.community/tensorflow/docs-l10n/site/en-snapshot/guide/migrate
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/AbstractRNNCell
# https://www.tensorflow.org/guide/keras/custom_layers_and_models/#layers_are_recursively_composable
# https://www.tensorflow.org/guide/function#creating_tfvariables
