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
class PPO(object):
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. 
    #The input states of RNN are different too. (For each step of RNN, input states are states of signal and the signal's chosen action.)

    def __init__(self, s_dim=32, num_intersection=1, a_dim = 2, name="meme", combine_action = 1, index = 0, with_FT = False, dic_traffic_env_conf = None, dic_agent_conf = None):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.num_intersection = num_intersection
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.dic_traffic_env_conf = dic_traffic_env_conf
            self.dic_agent_conf = dic_agent_conf
            self.with_FT = with_FT
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
            self.index = index

            ## 超参数
            global EP_MAX 
            EP_MAX = self.dic_agent_conf["EP_MAX"]
            global EP_LEN
            EP_LEN = self.dic_agent_conf["EP_LEN"]
            global GAMMA
            GAMMA = self.dic_agent_conf["GAMMA"]
            global A_LR
            A_LR = self.dic_agent_conf["A_LR"]
            global C_LR
            C_LR = self.dic_agent_conf["C_LR"]
            global BATCH
            BATCH = self.dic_agent_conf["BATCH"]
            global A_UPDATE_STEPS
            A_UPDATE_STEPS = self.dic_agent_conf["A_UPDATE_STEPS"]
            global C_UPDATE_STEPS
            C_UPDATE_STEPS = self.dic_agent_conf["C_UPDATE_STEPS"]
            

            # self.rnn_input = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)


            # with tf.variable_scope('rnn_input_cell'):
            #     Uw = tf.get_variable('Uw', [self.s_dim, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Ub = tf.get_variable('Ub', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_cycle_cell'):
            #     Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Wb = tf.get_variable('Wb', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_output_cell'):
            #     Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],initializer = tf.constant_initializer(0.0))
            #     Vb = tf.get_variable('Vb', [self.output_net], initializer=tf.constant_initializer(0.0))

            self.tfa = tf.placeholder(tf.int32, [None], 'action_{}'.format(self.index))
            self.tfadv = tf.placeholder(tf.float32, [None, int(self.num_intersection/self.combine_action)], 'advantage_{}'.format(self.index))
            self.tfs = tf.placeholder(tf.float32, [None, int(s_dim * self.combine_action/num_intersection)], 'actor_state_{}'.format(self.index))
            # critic
            with tf.variable_scope(self.name + '_critic_{}'.format(self.index)):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r_{}'.format(self.index))
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet_FCN(self.name + '_pi_{}'.format(self.index), trainable=True)
            self.oldpi, oldpi_params = self._build_anet_FCN(self.name + '_oldpi_{}'.format(self.index), trainable=False)

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

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
        
        ##Fiexed time 部分
        self.action = [0 for i in range(self.a_dim - 1)]
        self.current_phase_time = [0 for i in range(self.a_dim - 1)]
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == "anon":
            print("111111111111")
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        else:
            print("2222222222222222222222")
            self.DIC_PHASE_MAP = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                -1: -1
            }

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
        # self.summarize(actor_loss,self.global_counter,'Actor_loss_{}'.format(self.index))
        [self.sess.run(self.atrain_op, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        critic_loss = self.sess.run(self.closs, {self.tfs: c_s, self.tfdc_r: r})
        # self.summarize(critic_loss,self.global_counter,'Critic_loss_{}'.format(self.index))
        [self.sess.run(self.ctrain_op, {self.tfs:c_s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        self.global_counter += 1

        self.empty_buffer()

    def rnn_cell(self,rnn_input, state,name,trainable,last_prob):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        #Xt = [none,198 + 2] St-1 = [none,64] Yt = [none,64]
        #Uw = [198 + 2,64] Ub = [64]
        #Ww = [64,64]   Wb = [64]
        #Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable('Uw', [int(self.s_dim/self.num_intersection) + self.a_dim, self.hidden_net],trainable=trainable)
            Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
            Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
            Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        if last_prob == None:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.pad(rnn_input,[[0,0],[0,self.a_dim]]),[-1,int(self.s_dim/self.num_intersection) + self.a_dim]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        else:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.concat([tf.reshape(rnn_input,[-1,int(self.s_dim/self.num_intersection)]),last_prob],axis = 1),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))

        # St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.concat([rnn_input,last_prob],1),[-1,int(self.s_dim/self.num_intersection)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(tf.matmul(tf.cast(St,tf.float32),tf.cast(Vw,tf.float32)) + tf.cast(Vb,tf.float32))
        # return
        return St,Yt

    def _build_anet_FCN(self,name,trainable):
        with tf.variable_scope(name):
            input = tf.reshape(self.tfs,[-1,int(self.s_dim/self.num_intersection)])
            l1 = tf.layers.dense(input, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            # l2 = tf.layers.dense(l1, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
            #     kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            out = tf.layers.dense(l1, self.a_dim,tf.nn.softmax,trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out,params

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable('Uw', [int(self.s_dim/self.num_intersection) + self.a_dim, self.hidden_net],trainable=trainable)
                Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
                Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
                Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

            # RNN
            out_temp1 = []
            out_temp2 = []
            out = []
            actions = []
            last_prob = None
            rnn_input = tf.reshape(self.tfs,[-1,self.num_intersection,int(self.s_dim/self.num_intersection)])
            state = np.zeros([1,self.hidden_net])
            #The first for cycle aims to get state include all signals' imformation
            #and   pass to the second RNN layer (through variate "state")
            for j in range(self.num_intersection):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp1.append(tf.layers.dense(y, self.a_dim,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp1[j]
            #The second cycle is aim to make actions depend on last cycle's final state.
            last_prob = None
            for j in range(self.num_intersection):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp2.append(tf.layers.dense(y, self.a_dim,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp2[j]
                # actions = np.random.choice(range(out_temp1[j]),p=out_temp1[j].ravel())
            out = tf.stack([out_temp2[k] for k in range(self.num_intersection)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def calculate_FT_offset(self, state, action):
        if self.with_FT == True:
            for i in range(self.a_dim - 1):
                if state[i]["cur_phase"][0] == -1:
                    self.action[i] = self.action[i]
                cur_phase = self.DIC_PHASE_MAP[state[i]["cur_phase"][0]]
                #print(state)
                # print(state["time_this_phase"][0], self.dic_agent_conf["FIXED_TIME"][cur_phase], cur_phase)

                if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                    if state[i]["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                        self.current_phase_time[i] = 0
                        self.action[i] = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
                    else:
                        self.action[i] = cur_phase
                        self.current_phase_time[i] += 1
                else:
                    if state[i]["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                        self.current_phase_time[i] = 0
                        self.action[i] = 1
                    else:
                        self.current_phase_time[i] += 1
                        self.action[i] = 0
        ## agent调整FT动作
        action = action[0]
        if action < self.a_dim - 1:
            cur_phase = self.DIC_PHASE_MAP[state[action]["cur_phase"][0]]
            self.current_phase_time[action] = 0
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                if self.name == "HRC":
                    if state[action]["time_this_phase"][0] < self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                        self.action[action] = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
                else:
                    if cur_phase != -1:
                        self.action[action] = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
            else:
                self.action[action] = (cur_phase + 1) % 2
        return self.action

    def reset(self):
        print("\033[1;34m [---AGENT---]  Agent reset  \033[0m")
        self.action = [0 for i in range(self.a_dim - 1)]
        self.current_phase_time = [0 for i in range(self.a_dim - 1)]

    def choose_action(self, s):

        _s = np.array(s).reshape([-1,int(self.s_dim* self.combine_action/self.num_intersection)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,self.a_dim])
        # print(prob)

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


    def trajction_process_proximate(self):
        #This function aims to calculate proximate F(s) of each state s.
        v_s_ = np.mean(np.array(self.buffer_r).reshape([-1,self.num_intersection]),axis = 0)
        #we assume that all of the following Rs are the mean value of simulated steps (200)
        #so, the following Rs are geometric progression.
        #Sn = a1 * (1 - GAMMA^^n) / (1 - GAMMA) proximate equals to a1/(1-GAMMA)
        # print(v_s_)
        v_s_ = v_s_ / (1 - GAMMA)
        # print(v_s_)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.num_intersection])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.num_intersection):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.num_intersection):
            buffer[i].reverse()
        # print(np.array(buffer[0]))
        out = np.stack([buffer[k] for k in range(self.num_intersection)], axis=1)

        self.buffer_r = np.array(out).reshape([-1])


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