import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import time

latent_dim = 10

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.8
A_LR = 0.0005
C_LR = 0.0005
BATCH =50
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

###################Two cycle layer RNN###########################
#PPO3 has two layers of RNN neural network.
#First layer doesn't output actions and we record the last step's hidden state
#as the second cycle layer's first step's hidden input state.(Ensuring that all actions
#  decided at each timestep are depanded on all signals' states.)
class Cenlight(object):
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. 
    #The input states of RNN are different too. (For each step of RNN, input states are states of signal and the signal's chosen action.)

    def __init__(self, s_dim=32, num_intersection=1, a_dim = 2, name="meme", combine_action = 1):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.num_intersection = num_intersection
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64
            self.combine_action = combine_action
            self.lstm_layer = 1

            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, int(self.num_intersection/self.combine_action)], 'advantage')
            self.tfs = tf.placeholder(tf.float32, [None, int(s_dim * self.combine_action/num_intersection)], 'actor_state')
            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
            # actor
            self.pi = None 
            pi_params = None
            self.oldpi = None
            oldpi_params = None
            if name == "Cenlight":
                self.pi, pi_params = self._build_anet_cenlight_lstm(self.name + '_pi', trainable=True)
                self.oldpi, oldpi_params = self._build_anet_cenlight_lstm(self.name + '_oldpi', trainable=False)
            elif name == "RNN_Cenlight":
                self.pi, pi_params = self._build_anet_cenlight_rnn(self.name + '_pi', trainable=True)
                self.oldpi, oldpi_params = self._build_anet_cenlight_rnn(self.name + '_oldpi', trainable=False)
            elif name == "Bi_Cenlight":
                self.pi, pi_params = self._build_anet_bi_cenlight_lstm(self.name + '_pi', trainable=True)
                self.oldpi, oldpi_params = self._build_anet_bi_cenlight_lstm(self.name + '_oldpi', trainable=False)
            elif name == "Double_Cenlight":
                self.pi, pi_params = self._build_anet_double_cenlight_lstm(self.name + '_pi', trainable=True)
                self.oldpi, oldpi_params = self._build_anet_double_cenlight_lstm(self.name + '_oldpi', trainable=False)
            else:
                assert "\033[1;34m Undefined model name: {} \033[0m".format(self.model_name)
                
            new_params = []
            for p, oldp in zip(pi_params, oldpi_params):
                try:
                    new_params.append(oldp.assign(p))
                    print('\033[1;34m [---COPY PARAMS---] Assign [', p, '] to [', oldp, '] success \033[0m')
                except AttributeError:
                    print('\033[1;31m [---COPY PARAMS---] Assign [', p, '] to [', oldp, '] failed \033[0m')
            self.update_oldpi_op = new_params
            ##调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi,[-1,self.a_dim])
            self.oldpi_resize = tf.reshape(self.oldpi,[-1,self.a_dim])
            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=self.oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(tf.reduce_mean(tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.combine_action]),axis= 1),
                                                        [-1,int(self.num_intersection/self.combine_action)])
            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss
            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        ##以下为分开计算actor loss的部分
            self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.num_intersection]),axis = 0)
            # self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(self.num_intersection)]
            self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(1)]
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/PPO3/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=5)
            # tf.get_default_graph().finalize()

    def update(self, state):
        self.trajction_process(state)

        print("Update")
        s = np.vstack(self.buffer_s)
        c_s = s.reshape([-1,int(self.s_dim * self.combine_action/ self.num_intersection)])
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1])
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: c_s, self.tfdc_r: r})
        ##Calculating advantages one
        adv_r = np.array(adv).reshape([-1,int(self.num_intersection/self.combine_action)])
        actor_loss = self.sess.run(self.aloss, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r})
        # self.summarize(actor_loss,self.global_counter,'Actor_loss')
        [self.sess.run(self.atrain_op, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        critic_loss = self.sess.run(self.closs, {self.tfs: c_s, self.tfdc_r: r})
        # self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs:c_s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        self.global_counter += 1

        self.empty_buffer()
    
    # def lstm_cell(self, xt, last_ht, last_ct, name, trainable):
    #     ### ft = sigmoid(wf*[ht-1, xt] + bf)
    #     with tf.variable_scope('lstm_ft_' + name, reuse=True):
    #         wf = tf.get_variable('Wf', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
    #         bf = tf.get_variable('Bf', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)
    #     ### it = sigmoid(wi*[ht-1, xt] + bi)
    #     with tf.variable_scope('lstm_it_' + name, reuse=True):
    #         wi = tf.get_variable('Wi', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
    #         bi = tf.get_variable('Bi', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      
    #     ### _ct =  tanh(wc*[ht-1, xt] + bc)
    #     with tf.variable_scope('lstm_ct_' + name, reuse=True):
    #         wc = tf.get_variable('Wc', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
    #         bc = tf.get_variable('Bc', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      
    #     ### ot = sigmoid(wo*[ht-1, xt] + bo)
    #     ### ht = ot * tanh(Ct)
    #     with tf.variable_scope('lstm_ct_' + name, reuse=True):
    #         wo = tf.get_variable('Wo', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
    #         bo = tf.get_variable('Bo', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      

    #     cur_state = tf.concat([last_ht, xt], 0)
    #     ft = tf.nn.sigmoid(tf.matmul(cur_state, wf) + bf)
    #     it = tf.nn.sigmoid(tf.matmul(cur_state, wi) + bi)
    #     _ct = tf.nn.tanh(tf.matmul(cur_state, wc) + bc)
    #     Ct = tf.add(tf.multiply(ft, last_ct), tf.multiply(it, _ct))
    #     ot = tf.nn.sigmoid(tf.matmul(cur_state, wo) + bo)
    #     ht = tf.multiply(ot, tf.nn.tanh(Ct))
    #     return ht, Ct

        
    def _build_anet_cenlight_lstm(self, name, trainable):
        with tf.variable_scope(name):
            # ### ft = sigmoid(wf*[ht-1, xt] + bf)
            # with tf.variable_scope('lstm_ft_' + name):
            #     wf = tf.get_variable('Wf', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
            #     bf = tf.get_variable('Bf', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)
            # ### it = sigmoid(wi*[ht-1, xt] + bi)
            # with tf.variable_scope('lstm_it_' + name):
            #     wi = tf.get_variable('Wi', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
            #     bi = tf.get_variable('Bi', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      
            # ### _ct =  tanh(wc*[ht-1, xt] + bc)
            # with tf.variable_scope('lstm_ct_' + name):
            #     wc = tf.get_variable('Wc', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
            #     bc = tf.get_variable('Bc', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      
            # ### ot = sigmoid(wo*[ht-1, xt] + bo)
            # ### ht = ot * tanh(Ct)
            # with tf.variable_scope('lstm_ct_' + name):
            #     wo = tf.get_variable('Wo', [self.hidden_net, self.hidden_net],trainable=trainable, dtype=tf.float32)
            #     bo = tf.get_variable('Bo', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable, dtype=tf.float32)      
            
            weights = {
                'in_{}'.format(name): tf.Variable(tf.random_normal([int(self.s_dim / self.num_intersection), self.hidden_net]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.random_normal([self.hidden_net, self.a_dim]), trainable=trainable)
            }
            biases = {
                'in_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.hidden_net, ]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.a_dim, ]), trainable=trainable)
            }

            w_in = weights['in_{}'.format(name)]
            b_in = biases['in_{}'.format(name)]
            input = tf.reshape(self.tfs, [-1, int(self.s_dim / self.num_intersection)])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, self.num_intersection, self.hidden_net])
            tf.nn.rnn_cell
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_net, trainable=trainable, name=name) for i in range(self.lstm_layer)])
            # init_state = cell.zero_state(batch, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(fw_cell, input_rnn, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, self.hidden_net])
            w_out = weights['out_{}'.format(name)]
            b_out = biases['out_{}'.format(name)]
            pred = tf.nn.softmax(tf.matmul(output, w_out) + b_out)
            # out = tf.stack([pred[k] for k in range(self.num_intersection)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pred, params

        #     prob_output = []
        #     last_ht = tf.zeros([1,self.hidden_net], dtype=tf.float32)
        #     last_ct = tf.zeros([1,self.hidden_net], dtype=tf.float32)
        #     for j in range(self.num_intersection):
        #         last_ht, last_ct = self.lstm_cell(input_rnn[:,j,:], last_ht, last_ct, name, trainable)
        #         prob_output.append(tf.layers.dense(last_ht, self.a_dim, tf.nn.softmax, trainable = trainable, 
        #             kernel_initializer = tf.random_normal_initializer(0., .01),
        #             bias_initializer = tf.constant_initializer(0.01)))
        #     prob_output = tf.stack([prob_output[k] for k in range(self.num_intersection)], axis=1)
        # params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        # return prob_output, params

    def _build_anet_cenlight_rnn(self, name, trainable):
        with tf.variable_scope(name):
            weights = {
                'in_{}'.format(name): tf.Variable(tf.random_normal([int(self.s_dim / self.num_intersection), self.hidden_net]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.random_normal([self.hidden_net, self.a_dim]), trainable=trainable)
            }
            biases = {
                'in_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.hidden_net, ]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.a_dim, ]), trainable=trainable)
            }

            w_in = weights['in_{}'.format(name)]
            b_in = biases['in_{}'.format(name)]
            input = tf.reshape(self.tfs, [-1, int(self.s_dim / self.num_intersection)])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, self.num_intersection, self.hidden_net])
            tf.nn.rnn_cell
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(self.hidden_net, trainable=trainable, name=name) for i in range(self.lstm_layer)])
            # init_state = cell.zero_state(batch, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(fw_cell, input_rnn, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, self.hidden_net])
            w_out = weights['out_{}'.format(name)]
            b_out = biases['out_{}'.format(name)]
            pred = tf.nn.softmax(tf.matmul(output, w_out) + b_out)
            # out = tf.stack([pred[k] for k in range(self.num_intersection)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pred, params

    def _build_anet_bi_cenlight_lstm(self, name, trainable):
        with tf.variable_scope(name):    
            weights = {
                'in_{}'.format(name): tf.Variable(tf.random_normal([int(self.s_dim / self.num_intersection), self.hidden_net]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.random_normal([self.hidden_net, self.a_dim]), trainable=trainable)
            }
            biases = {
                'in_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.hidden_net, ]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.a_dim, ]), trainable=trainable)
            }

            w_in = weights['in_{}'.format(name)]
            b_in = biases['in_{}'.format(name)]
            input = tf.reshape(self.tfs, [-1, int(self.s_dim / self.num_intersection)])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, self.num_intersection, self.hidden_net])
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_net, trainable=trainable, name=name) for i in range(self.lstm_layer)])
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_net, trainable=trainable, name=name) for i in range(self.lstm_layer)])
            output_rnn, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_rnn, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, self.hidden_net])
            w_out = weights['out_{}'.format(name)]
            b_out = biases['out_{}'.format(name)]
            pred = tf.nn.softmax(tf.matmul(output, w_out) + b_out)
            # out = tf.stack([pred[k] for k in range(self.num_intersection)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pred, params
    
    def _build_anet_double_cenlight_lstm(self, name, trainable):
        with tf.variable_scope(name):
            weights = {
                'in_{}'.format(name): tf.Variable(tf.random_normal([int(self.s_dim / self.num_intersection), self.hidden_net]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.random_normal([self.hidden_net, self.a_dim]), trainable=trainable)
            }
            biases = {
                'in_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.hidden_net, ]), trainable=trainable),
                'out_{}'.format(name): tf.Variable(tf.constant(0.1, shape=[self.a_dim, ]), trainable=trainable)
            }

            w_in = weights['in_{}'.format(name)]
            b_in = biases['in_{}'.format(name)]
            input = tf.reshape(self.tfs, [-1, int(self.s_dim / self.num_intersection)])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, self.num_intersection, self.hidden_net])
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_net, trainable=trainable, name=name) for i in range(self.lstm_layer)])
            output_rnn, final_states = tf.nn.dynamic_rnn(fw_cell, input_rnn, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(fw_cell, input_rnn, initial_state = final_states, dtype=tf.float32 )
            output = tf.reshape(output_rnn, [-1, self.hidden_net])
            w_out = weights['out_{}'.format(name)]
            b_out = biases['out_{}'.format(name)]
            pred = tf.nn.softmax(tf.matmul(output, w_out) + b_out)
            # out = tf.stack([pred[k] for k in range(self.num_intersection)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pred, params

    def choose_action(self, s):
        _s = np.array(s).reshape([-1,int(self.s_dim* self.combine_action/self.num_intersection)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,self.a_dim])
        for i in range(self.num_intersection):
            action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                p=prob_temp[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)
        
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim * self.combine_action/self.num_intersection)]).tolist()
        v_s_ = self.get_v(_s)
        buffer_r = np.mean(np.array(self.buffer_r).reshape([-1,self.combine_action]), axis= 1).reshape([-1,int(self.num_intersection / self.combine_action)])
        buffer = [[] for i in range(self.num_intersection)]
        for r in buffer_r[::-1]:
            for i in range(int(self.num_intersection/self.combine_action)):
                v_s_[i] = (r[i] + GAMMA * v_s_[i])
                buffer[i].append(copy.deepcopy(v_s_[i]))
        for i in range(int(self.num_intersection/self.combine_action)):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(int(self.num_intersection/self.combine_action))], axis=1)
        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")