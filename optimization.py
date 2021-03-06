# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
tf.compat.v1.disable_resource_variables()
tf.compat.v1.disable_eager_execution()

try:
  import horovod.tensorflow as hvd
except:
  hvd = None

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, use_hvd=False, optimizer_type="adam"):
  """Creates an optimizer training op."""
  global_step = tf.compat.v1.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear or square root decay of the learning rate.
  if optimizer_type == "adam":
    power = 1.0
  elif optimizer_type == "lamb":
    power = 0.5
  elif optimizer_type == "nadam":
    power = 1.0
  else:
    power = 0.5
    
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=power,
      cycle=False)
  # if use_hvd:
  #   # May want to scale learning rate by number of GPUs
  #   learning_rate *= hvd.size()

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  if optimizer_type == "adam":
    print("Initializing ADAM Optimizer")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_type == "lamb":
    print("Initializing LAMB Optimizer")
    optimizer = LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_type == "nadam":
    print("Initializing NADAM Optimizer")
    optimizer = NadamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    print("Initializing NLAMB Optimizer")
    optimizer = NlambOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
      
  if use_hvd:
    # [HVD] Wrap the original optimizer by Horovod's distributed optimizer, which handles all the under the hood allreduce calls. 
    # Notice Horovod only does synchronized parameter update.
    optimizer = hvd.DistributedOptimizer(optimizer)

  if use_tpu:
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.compat.v1.trainable_variables()
  if use_hvd:
    # [HVD] Use distributed optimizer to compute gradients
    grads_and_vars=optimizer.compute_gradients(loss, tvars)
    grads = [grad for grad,var in grads_and_vars]
    tvars = [var for grad,var in grads_and_vars]
  else:
    # Use standard TF gradients
    grads = tf.gradients(ys=loss, xs=tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op

class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
        tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
        tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
          m.assign(next_m),
          v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.compat.v1.train.Optimizer):
  """
  LAMBOptimizer optimizer. 
  https://github.com/ymcui/LAMB_Optimizer_TF

  # References
  - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805

  # Parameters
  - There is nothing special, just the same as `AdamWeightDecayOptimizer`.
  """

  def __init__(self,
              learning_rate,
              weight_decay_rate=0.01,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-6,
              exclude_from_weight_decay=None,
              name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/lamb_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/lamb_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

      # Note: Here are two choices for scaling function \phi(z)
      # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
      # identity: \phi(z) = z
      # The authors does not mention what is \gamma_l and \gamma_u
      # UPDATE: after asking authors, they provide me the code below.
      # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
      #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      r1 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(param)))
      r2 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(update)))

      r = tf.compat.v1.where(tf.greater(r1, 0.0), tf.compat.v1.where(
        tf.greater(r2, 0.0), r1/r2, 1.0), 1.0)

      eta = self.learning_rate * r

      update_with_lr = eta * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
class NadamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """
  Optimizer that implements the Nadam algorithm.  Nadam is Adam with
  Nesterov momentum.
  
  A basic Nadam optimizer that includes "correct" L2 weight decay.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
    https://github.com/tdozat/Optimization/blob/master/tensorflow/nadam.py
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam
  """

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.00,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="NadamWeightDecayOptimizer"):
    """Constructs a NadamWeightDecayOptimizer."""
    super(NadamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    # get the local step 
    steps = tf.cast(global_step, tf.float32) + 1.
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      
      m = tf.compat.v1.get_variable(
          name=param_name + "/nadam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/nadam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))
     
      # We could use momentum scheduling variable 
      # mu_t   = self.beta_1 * (1. - 0.5 * (0.96**(0.004*steps))) 
      # instead we use constant scheduling so mu_t = self_beta1       
     
      beta1_correction = 1./(1. - (self.beta_1 ** steps))
      beta1_correction_tp1 = 1./(1. - (self.beta_1 ** (steps+1)))
      beta2_correction = 1./(1. - (self.beta_2 ** steps))

      next_m_unbiased = tf.multiply(beta1_correction_tp1,next_m)
      next_v_unbiased = tf.multiply(beta2_correction,next_v)
      # Nesterov addition moment calculation
      
      next_m_nesterov = (tf.multiply(self.beta_1, next_m_unbiased) + tf.multiply((1.0-self.beta_1)*beta1_correction,grad))

      update = next_m_nesterov / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
          m.assign(next_m),
          v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

class NlambOptimizer(tf.compat.v1.train.Optimizer):
  """
  Optimizer that implements the NLAMB algorithm.  Nlamb is Lamb with
  Nesterov momentum.
  
  A basic Nlamb optimizer that includes "correct" L2 weight decay.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
     https://github.com/tdozat/Optimization/blob/master/tensorflow/nadam.py
     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam
  """
  
  def __init__(self,
              learning_rate,
              weight_decay_rate=0.00,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-6,
              exclude_from_weight_decay=None,
              name="NlambOptimizer"):
    """Constructs a NlamOptimizer."""
    super(NlambOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    # get the local step 
    steps = tf.cast(global_step, tf.float32) + 1.
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/nlamb_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/nlamb_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
     
      beta1_correction     = 1./(1. - (self.beta_1 ** steps))
      beta1_correction_tp1 = 1./(1. - (self.beta_1 ** (steps+1)))
      beta2_correction     = 1./(1. - (self.beta_2 ** steps))

      next_m_unbiased = tf.multiply(beta1_correction_tp1,next_m)
      next_v_unbiased = tf.multiply(beta2_correction,next_v)
      # Nesterov addition moment calculation
      
      next_m_nesterov = (tf.multiply(self.beta_1, next_m_unbiased) + tf.multiply((1.0-self.beta_1)*beta1_correction,grad))
    
      update = next_m_nesterov / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

      # Note: Here are two choices for scaling function \phi(z)
      # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
      # identity: \phi(z) = z
      # The authors does not mention what is \gamma_l and \gamma_u
      # UPDATE: after asking authors, they provide me the code below.
      # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
      #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      r1 = tf.sqrt(tf.reduce_sum(tf.square(param)))
      r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

      r = tf.where(tf.greater(r1, 0.0), tf.where(
        tf.greater(r2, 0.0), r1/r2, 1.0), 1.0)

      eta = self.learning_rate * r

      update_with_lr = eta * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
