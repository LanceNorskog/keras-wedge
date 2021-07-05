import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import smart_cond

def wedge_random_channels_norm_batchwise(similarity, seed, inputs_flat, actual_batchsize, fmap_count, fmap_size, batchwise=True):
    fmap_even = (fmap_count//2)*2
    flatshape = (-1, fmap_size, fmap_count)
    indices = tf.constant(np.arange(fmap_count), dtype='int32')
    random_indices = tf.random.shuffle(indices, seed=seed)
    # print('random_indices:', random_indices)

    inputs_min = tf.math.reduce_min(inputs_flat, axis=1, keepdims=True)
    inputs_max = tf.math.reduce_max(inputs_flat, axis=1, keepdims=True)
    inputs_norm = (inputs_flat - inputs_min) / (inputs_max - inputs_min)
    # print('inputs_norm:', inputs_norm)
    # return inputs_norm
    mult_list = []
    for fm in range(0, fmap_even, 2):
        mult_list.append(inputs_norm[:, :, random_indices[fm]] * inputs_norm[:, :, random_indices[fm + 1]])
    mult = tf.stack(mult_list, axis=2)

    gam = tf.math.reduce_mean(mult, axis=1, keepdims=True)
    # print('gam:', gam)
    # return gam
    live = tf.cast(mult[:, :, :] > gam, dtype='float32')
    # print('live:', live)
    # return live
    percent = tf.math.reduce_sum(live, axis=1, keepdims=True) / flatshape[1]
    # print('percent:', percent)
    # return percent
    mask = tf.cast(percent < similarity, dtype='float32')
    # print('mask:', mask)
    # return mask
    if batchwise:
        mask = tf.cast(tf.math.reduce_mean(mask, axis=0) > 0.50001, dtype='float32')
        # I have no idea why these two variations are necessary
        if fmap_count > 3:
            mask = tf.squeeze(mask)
        else:
            mask = tf.reshape(mask, (1,))
        mask = tf.tile(mask, actual_batchsize)
        mask = tf.reshape(mask, (-1, 1, fmap_even // 2))
        print('batchwise')
    else:
        print('not batchwise')
    mask_unstack = tf.unstack(mask, axis=2)
    mask_list = []
    for i in range(fmap_even // 2):
        mask_list.append(mask_unstack[i])
        mask_list.append(mask_unstack[i])
    if fmap_even < fmap_count:
        mask_list.append(tf.ones_like(mask_unstack[0], dtype='float32'))
    dup_mask = tf.stack(mask_list, axis=1)
    # print('dup_mask:', dup_mask)
    # return dup_mask
    tiled_indices_flat = tf.tile(random_indices, actual_batchsize)
    tiled_indices = tf.reshape(tiled_indices_flat, (-1, 1, fmap_count))
    # print('tiled_indices:', tiled_indices)
    # return tiled_indices
    inverse_mask_flat = tf.gather(dup_mask, tiled_indices, batch_dims=1, axis=1)
    inverse_mask = tf.reshape(inverse_mask_flat, (-1, 1, fmap_count))
    # print('inverse_mask:', inverse_mask)
    # return inverse_mask
    return inputs_flat * inverse_mask

class WedgeDropout1D(tf.keras.layers.Layer):
    """Applies Wedge Dropout to the input.
    The WedgeDropout1D layer compares randomly chosen pairs of feature maps, and sets
    both feature maps to zero when they are "too similar". Similarity is measured
    by counting the pixels in both feature maps that are greater than the mean 
    of each feature map.
    When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.
    (This is in contrast to setting `trainable=False` for a WedgeDropout layer.
    `trainable` does not affect the layer's behavior, as WedgeDropout does
    not have any variables/weights that can be frozen during training.)
    >>> layer = WedgeDropout1D(.2, seed=0, input_shape=(2, 5, 2))
    >>> data = np.arange(20).reshape(2, 5, 2).astype(np.float32)
    >>> print(data)
    [[[ 0.  1.]
    [ 2.  3.]
    [ 4.  5.]
    [ 6.  7.]
    [ 8.  9.]]
  
   [[10. 11.]
    [12. 13.]
    [14. 15.]
    [16. 17.]
    [18. 19.]]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)
    Arguments:
      rate: Float between 0 and 1. 1.0 - rate = percentage of matching values 
      which triggers a dropout event
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """
  
    def __init__(self, similarity, batchwise=True, seed=None, **kwargs):
        super(WedgeDropout1D, self).__init__(**kwargs)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
        self.similarity = similarity
        self.seed = seed
        self.batchwise = batchwise
        self.supports_masking = True
  
    def build(self, input_shape):
        shape = input_shape.as_list()
        self.fmap_count = shape[2]
        self.fmap_even = (shape[2]//2)*2
        self.fmap_shape = (shape[1], )
        self.built = True
  
    def call(self, inputs, training=None):
        if training is None:
            # do not activate in untrainable section of trainable model
            if self.trainable:
                training = K.learning_phase()
            else:
                training = False
        
        def identity_inputs():
            return inputs
  
        def dropped_inputs():
            shape = inputs.shape.as_list()
            actual_batchsize = tf.shape(inputs)[0:1]
            self.fmap_shape = [shape[1]]
            self.fmap_count = shape[2]
            outputs = wedge_random_channels_norm_batchwise(self.similarity, self.seed, inputs, actual_batchsize, 
                                 self.fmap_count, self.fmap_shape[0], batchwise=self.batchwise)
            return outputs
  
        output = smart_cond.smart_cond(training, dropped_inputs,
                                            identity_inputs)
        return output
  
    # why?
    def _get_noise_shape(self, inputs):
        input_shape = array_ops.shape(inputs)
        noise_shape = (input_shape[0], 1, 1)
        return noise_shape

    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        config = {
            'similarity': self.similarity,
            'batchwise': self.batchwise,
            'seed': self.seed
        }
        base_config = super(WedgeDropout1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WedgeDropout2D(tf.keras.layers.Layer):
    """Applies Wedge Dropout to the input.
    The WedgeDropout2D layer compares randomly chosen pairs of feature maps, and sets
    both feature maps to zero when they are "too similar". Similarity is measured
    by counting the pixels in both feature maps that are greater than the mean 
    of each feature map.
    When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.
    (This is in contrast to setting `trainable=False` for a WedgeDropout2D layer.
    `trainable` does not affect the layer's behavior, as WedgeDropout2D does
    not have any variables/weights that can be frozen during training.)
    >>> layer = WedgeDropout2D(.65, seed=0, input_shape=(2, 5, 2))
    >>> data = np.arange(20).reshape(2, 5, 2).astype(np.float32)
    >>> print(data)
    [[[ 0.  1.]
    [ 2.  3.]
    [ 4.  5.]
    [ 6.  7.]
    [ 8.  9.]]
  
   [[10. 11.]
    [12. 13.]
    [14. 15.]
    [16. 17.]
    [18. 19.]]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)
    Arguments:
      rate: Float between 0 and 1. 1.0 - rate = percentage of matching values 
      which triggers a dropout event
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """
  
    def __init__(self, similarity=0.65, batchwise=True, seed=None, **kwargs):
        super(WedgeDropout2D, self).__init__(**kwargs)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
        self.similarity = similarity
        self.seed = seed
        self.batchwise = batchwise
        self.supports_masking = True
  
    def build(self, input_shape):
        print('WedgeDropout2D.build: input_shape:', input_shape)
        shape = input_shape.as_list()
        self.fmap_count = shape[3]
        self.fmap_even = (shape[3]//2)*2
        self.fmap_shape = (shape[1], shape[2])
        self.built = True
  
    def call(self, inputs, training=None):
        if training is None:
            # do not activate in untrainable section of trainable model
            if self.trainable:
                training = K.learning_phase()
            else:
                training = False
        
        def identity_inputs():
            return inputs
  
        def dropped_inputs():
            shape = inputs.shape.as_list()
            actual_batchsize = tf.shape(inputs)[0:1]
            self.fmap_shape = [shape[1], shape[2]]
            self.fmap_count = shape[3]
            input_shape = (-1, self.fmap_shape[0], self.fmap_shape[1], self.fmap_count)
            flatshape = (-1, self.fmap_shape[0] * self.fmap_shape[1], self.fmap_count)
            inputs_flatmap = tf.reshape(inputs, flatshape)
            outputs_flat = wedge_random_channels_norm_batchwise(self.similarity, self.seed, inputs_flatmap, actual_batchsize, 
                                 self.fmap_count, self.fmap_shape[0] * self.fmap_shape[1], batchwise=self.batchwise)
            outputs = tf.reshape(outputs_flat, tf.shape(inputs))
            return outputs
  
        output = smart_cond.smart_cond(training, dropped_inputs,
                                            identity_inputs)
        return output
  
    # why?
    def _get_noise_shape(self, inputs):
        input_shape = array_ops.shape(inputs)
        noise_shape = (input_shape[0], 1, 1, 1)
        return noise_shape

    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        config = {
            'similarity': self.similarity,
            'batchwise': self.batchwise,
            'seed': self.seed
        }
        base_config = super(WedgeDropout2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WedgeDropout3D(tf.keras.layers.Layer):
    """Applies Wedge Dropout to the input.
    The WedgeDropout3D layer compares randomly chosen pairs of feature maps, and sets
    both feature maps to zero when they are "too similar". Similarity is measured
    by counting the pixels in both feature maps that are greater than the mean 
    of each feature map.
    When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.
    (This is in contrast to setting `trainable=False` for a WedgeDropout3D layer.
    `trainable` does not affect the layer's behavior, as WedgeDropout3D does
    not have any variables/weights that can be frozen during training.)
    >>> layer = WedgeDropout3D(.2, seed=0, input_shape=(2, 5, 2))
    >>> data = np.arange(20).reshape(2, 5, 2).astype(np.float32)
    >>> print(data)
    [[[ 0.  1.]
    [ 2.  3.]
    [ 4.  5.]
    [ 6.  7.]
    [ 8.  9.]]
  
   [[10. 11.]
    [12. 13.]
    [14. 15.]
    [16. 17.]
    [18. 19.]]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)
    Arguments:
      rate: Float between 0 and 1. 1.0 - rate = percentage of matching values 
      which triggers a dropout event
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """
  
    def __init__(self, similarity, batchwise=True, seed=None, **kwargs):
        super(WedgeDropout3D, self).__init__(**kwargs)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
        self.similarity = similarity
        self.seed = seed
        self.batchwise = batchwise
        self.supports_masking = True
  
    def build(self, input_shape):
        shape = input_shape.as_list()
        self.fmap_count = shape[4]
        self.fmap_even = (shape[4]//2)*2
        self.fmap_shape = (shape[1], shape[2], shape[3])
        self.built = True
  
    def call(self, inputs, training=None):
        if training is None:
            # do not activate in untrainable section of trainable model
            if self.trainable:
                training = K.learning_phase()
            else:
                training = False
        
        def identity_inputs():
            return inputs
  
        def dropped_inputs():
            shape = inputs.shape.as_list()
            actual_batchsize = tf.shape(inputs)[0:1]
            self.fmap_shape = [shape[1], shape[2], shape[3]]
            self.fmap_count = shape[4]
            input_shape = (-1, self.fmap_shape[0], self.fmap_shape[1], self.fmap_shape[2], self.fmap_count)
            flatshape = (-1, self.fmap_shape[0] * self.fmap_shape[1] * self.fmap_shape[2], self.fmap_count)
            inputs_flatmap = tf.reshape(inputs, flatshape)
            outputs_flat = wedge_random_channels_norm_batchwise(self.similarity, self.seed, inputs_flatmap, actual_batchsize, 
                                 self.fmap_count, self.fmap_shape[0] * self.fmap_shape[1], batchwise=self.batchwise)
            outputs = tf.reshape(outputs_flat, tf.shape(inputs))
            return outputs
  
        output = smart_cond.smart_cond(training, dropped_inputs,
                                            identity_inputs)
        return output
  
    # why?
    def _get_noise_shape(self, inputs):
        input_shape = array_ops.shape(inputs)
        noise_shape = (input_shape[0], 1, 1, 1, 1)
        return noise_shape

    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        config = {
            'similarity': self.similarity,
            'batchwise': self.batchwise,
            'seed': self.seed
        }
        base_config = super(WedgeDropout3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WedgeDropoutLSTM2D(tf.keras.layers.Layer):
    """Applies Wedge Dropout to the input.
    
    Conv2DLSTM output: (batch, time, row, col, fmaps)
    
    The WedgeDropoutLSTM2D layer compares randomly chosen pairs of feature maps, and sets
    both feature maps to zero when they are "too similar". Similarity is measured
    by counting the pixels in both feature maps that are greater than the mean 
    of each feature map.
    When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.
    (This is in contrast to setting `trainable=False` for a WedgeDropoutLSTM2D layer.
    `trainable` does not affect the layer's behavior, as WedgeDropoutLSTM2D does
    not have any variables/weights that can be frozen during training.)
    >>> layer = WedgeDropoutLSTM2D(.2, seed=0, input_shape=(2, 5, 2))
    >>> data = np.arange(20).reshape(2, 5, 2).astype(np.float32)
    >>> print(data)
    [[[ 0.  1.]
    [ 2.  3.]
    [ 4.  5.]
    [ 6.  7.]
    [ 8.  9.]]
  
   [[10. 11.]
    [12. 13.]
    [14. 15.]
    [16. 17.]
    [18. 19.]]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)
    Arguments:
      rate: Float between 0 and 1. 1.0 - rate = percentage of matching values 
      which triggers a dropout event
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """
  
    def __init__(self, similarity, batchwise=True, seed=None, training=False, **kwargs):
        super(WedgeDropoutLSTM2D, self).__init__(**kwargs)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
        self.similarity = similarity
        self.seed = seed
        self.batchwise = batchwise
        self.supports_masking = True
        self.training = training
  
    def build(self, input_shape):
        self.built = True
  
    def call(self, inputs, training=None):
        if training is None:
            # do not activate in untrainable section of trainable model
            if self.training or self.trainable:
                training = K.learning_phase()
            else:
                training = False
        
        def identity_inputs():
            return inputs
  
        # each timestep is a separate batch
        # transpose to make timesteps * fmaps the final dimension
        def dropped_inputs():
            shape = inputs.shape.as_list()
            actual_batchsize = tf.shape(inputs)[0:1]
            timesteps = shape[1]
            self.fmap_shape = [shape[2], shape[3]]
            self.fmap_count = shape[4]
            transposed = tf.transpose(inputs, perm=(0,2,3,1,4))
            flatshape = (-1, self.fmap_shape[0] * self.fmap_shape[1], timesteps * self.fmap_count)
            inputs_flatmap = tf.reshape(transposed, flatshape)
            outputs_flat = wedge_random_channels_norm_batchwise(self.similarity, self.seed, inputs_flatmap, actual_batchsize, 
                                 flatshape[2], flatshape[1], batchwise=self.batchwise)
            out_shape = (-1, self.fmap_shape[0], self.fmap_shape[1], timesteps, self.fmap_count)
            out_reshaped = tf.reshape(outputs_flat, out_shape)
            out_trans = tf.transpose(out_reshaped, perm=(0,3,1,2,4))
            outputs = tf.reshape(out_trans, tf.shape(inputs))
            return outputs
  
        output = smart_cond.smart_cond(training, dropped_inputs,
                                            identity_inputs)
        return output
  
    # why?
    def _get_noise_shape(self, inputs):
        input_shape = array_ops.shape(inputs)
        noise_shape = (input_shape[0], 1, 1, 1, 1)
        return noise_shape

    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        config = {
            'similarity': self.similarity,
            'batchwise': self.batchwise,
            'seed': self.seed
        }
        base_config = super(WedgeDropoutLSTM2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
