import tensorflow as tf
import tensorflow.contrib.slim as slim

def upsample_and_sum(x1, x2, output_channels, in_channels, scope=None):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02), name=scope)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = deconv + x2
    deconv_output.set_shape([None, None, None, output_channels])

    return deconv_output

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h / scale_factor_h, w / scale_factor_w]

    return tf.image.resize_bilinear(x, size=new_size)

def resBlock(x,name_scope,channels=64,kernel_size=[3,3],scale=1):

    with tf.variable_scope(name_scope):
        tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
        tmp = tf.nn.relu(tmp)
        tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
        tmp *= scale
    return x + tmp

def res_upsample_and_sum(x1, x2, output_channels, in_channels, scope=None):
    pool_size = 2
    x2 = resBlock(x2,scope,output_channels)
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02), name=scope)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = deconv + x2
    deconv_output.set_shape([None, None, None, output_channels])

    return deconv_output

def param_free_norm(x, epsilon=1e-5) :
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x_std = tf.sqrt(x_var + epsilon)

    return (x - x_mean) / x_std

def spade(noise_map, x_init, channels, scope='spade') :

    xinit_shape = tf.unstack(tf.shape(x_init))
    noise_map_down = tf.image.resize_bilinear(noise_map, [xinit_shape[1], xinit_shape[2]], name='downsampled')
    with tf.variable_scope(scope) :
        x = param_free_norm(x_init)
        tmp = tf.layers.conv2d(noise_map_down, 64, [5, 5],
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='conv1')
        tmp = tf.nn.relu(tmp)
        noisemap_gamma = tf.layers.conv2d(tmp, channels, [3, 3],
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='conv_gamma')
        noisemap_beta = tf.layers.conv2d(tmp, channels, [3, 3],
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                          padding='same',
                                          name='conv_beta')

        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x

def spade_resblock(noisemap, x_init, channels, scope='spade_resblock'):
    with tf.variable_scope(scope) :
        x = spade(noisemap, x_init,channels,  scope='spade_1')
        x = tf.nn.leaky_relu(x, 0.02)
        x = tf.layers.conv2d(x, channels, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv1')

        x = spade(noisemap, x, channels, scope='spade_2')
        x = tf.nn.leaky_relu(x, 0.02)
        x = tf.layers.conv2d(x, channels, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv2')

        return x + x_init

def FCN(input):
    with tf.variable_scope('fcn'):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
        conv4 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4')
        conv5 = slim.conv2d(conv4, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5')
        return conv5

def FCN_Avg(input):
    with tf.variable_scope('fcn'):
        x = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
        x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
        x = tf.layers.average_pooling2d(x, [4, 4], [4, 4], padding='same', name='pooling')
        x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
        x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4')
        x = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
        x = slim.conv2d(x, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5')
        image_shape = tf.unstack(tf.shape(input))
        y = tf.image.resize_images(x,[image_shape[1],image_shape[2]])
        y = slim.conv2d(y, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv6')
        y = slim.conv2d(y, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv7')
        return x,y

def FCN2(input):
    with tf.variable_scope('Tri_EST'):
        with tf.variable_scope('BackBone'):
            x = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')

        with tf.variable_scope('SigmaC_EST'):
            x2 = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling2')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling3')
            x2 = slim.conv2d(x2, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
            sigma_c = tf.reduce_mean(tf.reduce_mean(x2, axis=1, keep_dims=True), axis=2, keep_dims=True, name='sigma_c')
            # sigma_cmap = tf.multiply(sigma_c, tf.ones_like(input))

        with tf.variable_scope('Sigma_Total'):
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_2')
            x = slim.conv2d(x, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_1')

        return sigma_c,x

def FCN2_Avg(input):
    with tf.variable_scope('Tri_EST'):
        with tf.variable_scope('BackBone'):
            x = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')

        with tf.variable_scope('SigmaC_EST'):
            x2 = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling2')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling3')
            x2 = slim.conv2d(x2, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
            sigma_c = tf.reduce_mean(tf.reduce_mean(x2, axis=1, keep_dims=True), axis=2, keep_dims=True, name='sigma_c')
            # sigma_cmap = tf.multiply(sigma_c, tf.ones_like(input))

        with tf.variable_scope('Sigma_Total'):
            x = tf.layers.average_pooling2d(x, [4, 4], [4, 4], padding='same', name='pooling')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4')
            x = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
            x = slim.conv2d(x, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5')
            image_shape = tf.unstack(tf.shape(input))
            y = tf.image.resize_images(x, [image_shape[1], image_shape[2]])
            y = slim.conv2d(y, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv6')
            y = slim.conv2d(y, 3, [7, 7], rate=1, activation_fn=tf.nn.relu, scope='conv7')
            y = slim.conv2d(y, 3, [7, 7], rate=1, activation_fn=tf.nn.relu, scope='conv8')
            y = slim.conv2d(y, 3, [7, 7], rate=1, activation_fn=tf.nn.relu, scope='conv9')

        return sigma_c,x,y

def FCN2_test(input):
    with tf.variable_scope('Tri_EST'):
        with tf.variable_scope('BackBone'):
            x = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')

        with tf.variable_scope('SigmaC_EST'):
            x2 = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling2')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling3')
            x2 = slim.conv2d(x2, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
            x2 = tf.layers.average_pooling2d(x2, [4, 4], [4, 4], padding='same', name='pooling4')
        return x2

def FCN3(input):
    with tf.variable_scope('Tri_EST'):
        with tf.variable_scope('BackBone'):
            x = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')

        with tf.variable_scope('SigmaC_EST'):
            x2 = tf.layers.average_pooling2d(x, [2, 2], [2, 2], padding='same', name='pooling')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling2')
            x2 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
            x2 = tf.layers.average_pooling2d(x2, [2, 2], [2, 2], padding='same', name='pooling3')
            x2 = slim.conv2d(x2, 3, [3, 3], rate=1, activation_fn=None, scope='conv3')
            sigma_c = tf.reduce_mean(tf.reduce_mean(x2, axis=1, keep_dims=True), axis=2, keep_dims=True, name='sigma_c')
            # sigma_cmap = tf.multiply(sigma_c, tf.ones_like(input))

        with tf.variable_scope('Sigma_Total'):
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_1')
            x = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_2')
            x = slim.conv2d(x, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_1')

        return sigma_c,x

def SPADE_RESUNET(input,noise_map):
    with tf.variable_scope('SPAD_CBDNET'):
        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_1')
        conv1 = spade_resblock(noise_map,conv1,64,'spaderes1_1')
        conv1 = spade_resblock(noise_map,conv1,64,'spaderes1_2')

        pool1 = slim.avg_pool2d(conv1, [2, 2], padding='SAME')
        conv2 = slim.conv2d(pool1, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_1')
        conv2 = spade_resblock(noise_map,conv2,128,'spaderes2_1')
        conv2 = spade_resblock(noise_map,conv2,128,'spaderes2_2')


        pool2 = slim.avg_pool2d(conv2, [2, 2], padding='SAME')
        conv3 = slim.conv2d(pool2, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_1')
        conv3 = spade_resblock(noise_map,conv3,256,'spaderes3_1')
        conv3 = spade_resblock(noise_map,conv3,256,'spaderes3_2')
        conv3 = spade_resblock(noise_map,conv3,256,'spaderes3_3')
        conv3 = spade_resblock(noise_map,conv3,256,'spaderes3_4')
        conv3 = spade_resblock(noise_map,conv3,256,'spaderes3_5')


        up4 = res_upsample_and_sum(conv3, conv2, 128, 256, scope='deconv4')
        conv4 = spade_resblock(noise_map,up4,128,'spaderes4_1')
        conv4 = spade_resblock(noise_map,conv4,128,'spaderes4_2')
        conv4 = spade_resblock(noise_map,conv4,128,'spaderes4_3')

        up5 = res_upsample_and_sum(conv4, conv1, 64, 128, scope='deconv5')
        conv5 = spade_resblock(noise_map,up5,64,'spaderes5_1')
        conv5 = spade_resblock(noise_map,conv5,64,'spaderes5_2')
        out = slim.conv2d(conv5, 3, [1, 1], rate=1, activation_fn=None, scope='conv6')

        return out


def SPADE_RESCBDNET(input):
    noise_map = FCN2_test(input)
    image_shape = tf.unstack(tf.shape(input))
    noise_map = tf.image.resize_images(noise_map, [image_shape[1], image_shape[2]])
    out = SPADE_RESUNET(input,noise_map) + input

    return noise_map, out

def SPADE_RESCBDNET2(input):
    down_noise_map, noise_map = FCN_Avg(input)
    image_shape = tf.unstack(tf.shape(input))
    upsample_noise_map = tf.image.resize_images(down_noise_map, [image_shape[1], image_shape[2]])
    noise_map = 0.8 *upsample_noise_map + 0.2*noise_map
    out = SPADE_RESUNET(input,noise_map) + input

    return noise_map, out

def SPADE_RESCBDNET3(input):
    down_noise_map, noise_map = FCN_Avg(input)
    image_shape = tf.unstack(tf.shape(input))
    upsample_noise_map = tf.image.resize_images(down_noise_map, [image_shape[1], image_shape[2]])
    noise_map = 0.8 *upsample_noise_map + 0.2*noise_map
    out = SPADE_RESUNET(input,noise_map) + input

    return down_noise_map,upsample_noise_map, out