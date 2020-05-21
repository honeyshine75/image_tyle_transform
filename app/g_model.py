import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras

class ConditionalInstanceNormalization(keras.layers.Layer):
    """
    A Conditional Instance Normalization Layer Implementation
    """
    def __init__(self,
                 n_styles,
                 epsilon=1e-3,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

        self.n_styles = n_styles
        self.epsilon = epsilon
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        
        self.gamma = self.add_weight(shape=[self.n_styles, input_shape[0][-1]],
                                     name="gamma",
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.beta = self.add_weight(shape=[self.n_styles, input_shape[0][-1]],
                                    name="beta",
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

    def call(self, inputs):
        """
        inputs:
            image: incoming activation
            styles: a __column__ tensor (i.e. [N, 1]) of weights for individual styles.
        first normalizes the input image,
        and then multiplies it with a weighted sum of the gamma and beta parameters
        """

        image, gamma_weights, beta_weights = inputs

        mu, sigma_sq = tf.nn.moments(image, axes=[1, 2], keepdims=True)
        normalized = (image - mu) / tf.sqrt(sigma_sq + self.epsilon)
        
        gamma = tf.expand_dims(gamma_weights, -1) * self.gamma
        beta = tf.expand_dims(beta_weights, -1) * self.beta
        
        gamma = tf.reduce_sum(gamma, axis=1)
        beta = tf.reduce_sum(beta, axis=1)

        gamma = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)

        return gamma * normalized + beta

    def get_config(self):
        base_config = super(ConditionalInstanceNormalization,
                            self).get_config()
        base_config["n_styles"] = self.n_styles
        base_config["epsilon"] = self.epsilon
        base_config["beta_initializer"] = self.beta_initializer
        base_config["gamma_initializer"] = self.gamma_initializer
        base_config["beta_regularizer"] = self.beta_regularizer
        base_config["gamma_regularizer"] = self.gamma_regularizer
        base_config["beta_constraint"] = self.beta_constraint
        base_config["gamma_constraint"] = self.gamma_constraint
        return base_config
        

initializer = tf.random_normal_initializer(0., 0.01)

class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if s[1] == None:
            return (None, None, None, s[3])
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        print(config)
        return config
    

class Convolution(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='REFLECT', activation='relu', 
                             kernel_initializer=initializer, normal='BN', trainable=True, **kwargs):
        
        super(Convolution, self).__init__(**kwargs)
        
        self.padding = padding
        self.normal = normal
        self.filters = filters
        
        pad = int((kernel_size - 1) / 2)
        self.pad = ReflectionPadding2D(padding=(pad, pad))
        
        self.conv1 = keras.layers.Conv2D(filters, kernel_size, strides, padding='valid', 
                                         kernel_initializer=initializer, activation=activation, trainable=trainable)
        
        if normal == 'BN':
            self.bn1 = keras.layers.BatchNormalization()
        elif normal == 'CIN':
            self.bn1 = ConditionalInstanceNormalization(n_styles=filters)
            self.conv2 = keras.layers.Conv2D(filters, 1, 1, kernel_initializer=initializer)
            self.conv3 = keras.layers.Conv2D(filters, 1, 1, kernel_initializer=initializer)
        else:
            print('wrong normal input!')
        
    def call(self, input):
        if self.normal == 'CIN':
            input, input_2 = input
            if self.padding is not None:
                output = self.pad(input)
            else:
                output = input
            output = self.conv1(output)
            output_2 = self.conv2(input_2)
            output_3 = self.conv3(input_2)
            output_2 = keras.layers.Reshape((self.filters,))(output_2)
            output_3 = keras.layers.Reshape((self.filters,))(output_3)
            output = self.bn1([output, output_2, output_3])
        
        
        else:
            if self.padding is not None:
                output = self.pad(input)
            else:
                output = input
            output = self.conv1(output)
            output = self.bn1(output)

        return output
        

class ResidualBlock(tf.keras.Model):
    def __init__(self, num_channels, strides=1, trainable=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Convolution(num_channels, strides=strides, kernel_size=3, normal='CIN', trainable=trainable)
        self.conv2 = Convolution(num_channels, strides=strides, kernel_size=3, activation='linear', normal='CIN', trainable=trainable)

    def call(self, x):
        x_1, x_2 = x
        y = self.conv1([x_1, x_2])
        y = self.conv2([y, x_2])

        return (y + x_1)
        

class Upsampling(tf.keras.Model):
    def __init__(self, num_channels, trainable=True, **kwargs):
        super(Upsampling, self).__init__(**kwargs)
        
        self.up1 = keras.layers.UpSampling2D(size=2)
        self.conv1 = Convolution(num_channels, kernel_size=3, strides=1, normal='CIN', trainable=trainable)

    def call(self, x):
        x_1, x_2 =x
        y = self.up1(x_1)
        y = self.conv1([y, x_2])
        y = keras.activations.relu(y)

        return y
        


def get_Net(trainable=True):
    input = keras.layers.Input(shape=(None, None, 3))
    input_2 = keras.layers.Input(shape=(256, 256, 3))

    InceptionV3_path = '../static/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    InceptionV3 = keras.applications.InceptionV3(weights=InceptionV3_path, include_top=False)
    base_model = keras.Model(inputs=InceptionV3.input, outputs=InceptionV3.get_layer('mixed6').output)
    base_model.trainable = False
    out_2 = base_model(input_2, training=False)
    out_2 = tf.reduce_mean(out_2, axis=[1, 2], keepdims=True)
    out_2 = keras.layers.Conv2D(100, 1, 1, kernel_initializer=initializer)(out_2)
    # out_2.shape = [None,  1, 1, 100]
    
    x = input
    x = Convolution(filters=32, kernel_size=9, strides=1, activation='relu', trainable=trainable)(x)
    x = Convolution(filters=64, kernel_size=3, strides=2, activation='relu', trainable=trainable)(x)
    x = Convolution(filters=128, kernel_size=3, strides=2, activation='relu', trainable=trainable)(x)

    x = ResidualBlock(128, trainable=trainable)([x, out_2])
    x = ResidualBlock(128, trainable=trainable)([x, out_2])
    x = ResidualBlock(128, trainable=trainable)([x, out_2])
    x = ResidualBlock(128, trainable=trainable)([x, out_2])
    x = ResidualBlock(128, trainable=trainable)([x, out_2])
    
    x = Upsampling(64, trainable=trainable)([x, out_2])
    x = Upsampling(32, trainable=trainable)([x, out_2])
    
    output = Convolution(filters=3, kernel_size=9, strides=1, activation='sigmoid', normal='CIN', trainable=trainable)([x, out_2])
    
    model = keras.Model(inputs=[input, input_2], outputs=output)
    
    return model

def precession(content_path, style_path):
    content = tf.io.read_file(content_path)
    if content_path.lower().endswith('png'):
        content = tf.image.decode_png(content, channels=3)
    else:
        content = tf.image.decode_jpeg(content, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
    content = tf.image.convert_image_dtype(content, tf.float32)
    h, w, c = content.shape
    max_allow_h_w = 1080*1080
    if h*w > max_allow_h_w:
        beta = (max_allow_h_w / (h * w))**(1/2)
        new_h = int(h * beta)
        new_w = int(w * beta)
        content = tf.image.resize(content, size=[new_h, new_w])
    content = tf.expand_dims(content, axis=0)

    style = tf.io.read_file(style_path)
    if style_path.lower().endswith('png'):
        style = tf.image.decode_png(style, channels=3)
    else:
        style = tf.image.decode_jpeg(style, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
    style = tf.image.convert_image_dtype(style, tf.float32)
    style = tf.image.resize(style, size=[256, 256])
    style = tf.expand_dims(style, axis=0)

    return content, style