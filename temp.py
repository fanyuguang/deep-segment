import tensorflow as tf

inputs = tf.constant(list(range(25))[1:], shape=[2, 3, 4], dtype=tf.float32)
hidden_size = 4
attention_size = 5

w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
vu = tf.tensordot(v, u_omega, axes=1, name='vu')
alphas = tf.nn.softmax(vu, name='alphas')

temp1 = inputs * tf.expand_dims(alphas, -1)
output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    temp1_value, output_value = sess.run([temp1, output])
    print(temp1_value)
    print('')
    print(output_value)
