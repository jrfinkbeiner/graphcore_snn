import tensorflow as tf

class SparseLinear(tf.keras.layers.Layer):

  def __init__(self, inp_units, out_units):
      super().__init__()
      self.inp_units = inp_units
      self.out_units = out_units

  def build(self, input_shape):
      self.w = self.add_weight(shape=(self.inp_units, self.out_units),
                               initializer='random_normal',
                               trainable=True)
    #   self.b = self.add_weight(shape=(self.units,),
    #                            initializer='random_normal',
    #                            trainable=True)

  def call(self, inputs: tf.SparseTensor):
      return tf.sparse.sparse_dense_matmul(inputs, self.w) # + self.b