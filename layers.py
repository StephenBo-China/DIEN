import tensorflow as tf
from tensorflow.keras import layers

class GRU_GATES(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GRU_GATES, self).__init__()
        self.linear_act = layers.Dense(units, activation=None, use_bias=True)
        self.linear_noact = layers.Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))

class AUGRU(layers.Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()
        self.u_gate = GRU_GATES(units)
        self.r_gate = GRU_GATES(units)
        self.c_memo = GRU_GATES(units)

    def call(self, inputs, state, att_score):
        u = self.u_gate(inputs, state) #u_t
        r = self.r_gate(inputs, state) #r_t
        c = self.c_memo(inputs, state, r) #\tilde{h_t}
        u_= att_score * u #\tilde{u_{t}'} [AUGRU Add]
        state_next = (1 - u_) * state + u_ * c #h_t [AUGRU change u_t on output]
        return state_next