
""" Extensions to TF RNN class by una_dinosaria"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs

import collections
import math
import tensorflow as tf

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Add the residual connection
    output = tf.add(output, inputs)

    return output, new_state

class LinearSpaceDecoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    self.w_out = tf.get_variable("proj_w_out",
        [insize, output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    self.b_out = tf.get_variable("proj_b_out", [output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    self.linear_output_size = output_size


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size

  def __call__(self, inputs, state, scope=None):
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Apply the multiplication to everything
    output = tf.matmul(output, self.w_out) + self.b_out

    return output, new_state
