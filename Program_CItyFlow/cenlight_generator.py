from ppo_agent import PPO
from anon_env import AnonEnv
import os
import time
import numpy as np
# from test_cenlight_agent import Cenlight
from cenlight_agent import Cenlight

class cenlight_generator(object):
    def __init__(self,dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.num_intersection = dic_traffic_env_conf["NUM_INTERSECTIONS"]
        self.state_dim = self.compute_len_feature() * self.num_intersection
        self.action_dim = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.model_name = self.dic_agent_conf["AGENT_NAME"]
        print("\033[1;34m [---CONFIG---]  Model name: ", self.model_name, "\033[0m")
        print("\033[1;34m [---CONFIG---] State dim: ",self.state_dim, "\033[0m")
        print("\033[1;34m [---CONFIG---] Num of intersection: ", self.num_intersection, "\033[0m")
        print("\033[1;34m [---CONFIG---] Action dim: ",self.action_dim, "\033[0m")
        print("\033[1;34m [---CONFIG---] State list : ", self.dic_traffic_env_conf["LIST_STATE_FEATURE"], "\033[0m")
        print("\033[1;34m [---CONFIG---] State dim list : ", self.dic_traffic_env_conf["DIC_FEATURE_DIM"], "\033[0m")
        if self.model_name == "Cenlight" or self.model_name == "RNN_Cenlight" or self.model_name == "Bi_Cenlight" or self.model_name == "Double_Cenlight":
            self.agent = Cenlight(s_dim=self.state_dim, num_intersection=self.num_intersection, a_dim=self.action_dim, name=self.model_name)
        elif self.model_name == "Single_PPO":
            self.agent = PPO(s_dim=self.state_dim, num_intersection=1, a_dim=self.num_intersection + 1, name=self.model_name, dic_traffic_env_conf = self.dic_traffic_env_conf, dic_agent_conf = self.dic_agent_conf)
        elif self.model_name == "HRC":
            self.agent = PPO(s_dim=self.state_dim, num_intersection=1, a_dim=self.num_intersection + 1, name=self.model_name, with_FT = True, dic_traffic_env_conf = self.dic_traffic_env_conf, dic_agent_conf = self.dic_agent_conf)
        elif self.model_name == "Multi_PPO" or  self.model_name == "Multi_PPO_Neighbor":
            self.agent = [PPO(s_dim=self.compute_len_feature(), num_intersection=1, a_dim=self.action_dim, name=self.model_name, index= i) for i in range(self.num_intersection)]
        else:
            assert "\033[1;34m Undefined model name: {} \033[0m".format(self.model_name)
    
    ### anon tools
    def compute_len_feature(self):
        from functools import reduce
        num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature_name=="lane_num_vehicle":
                print("(self.dic_traffic_env_conf[DIC_FEATURE_DIM][D_+feature_name.upper()][0]*num_lanes,)", (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*num_lanes,))
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*num_lanes,)
            elif "phase" in feature_name:
                print("self.dic_traffic_env_conf[DIC_FEATURE_DIM][D_+feature_name.upper()]", self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()])
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            else:
                print("feature_name", feature_name)
                print("self.dic_traffic_env_conf[DIC_FEATURE_DIM][D_+feature_name.upper()]", self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()])
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
        return sum(len_feature)

    def dict_to_list(self, dict_state):
        list_state = [[] for i in range(self.num_intersection)]
        for i in range(self.num_intersection):
            for j in dict_state[i].keys():
                cur_state = dict_state[i][j]
                if 'phase' in j and j != 'time_this_phase':
                    cur_state = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][dict_state[i][j][0]]
                list_state[i] = np.append(list_state[i], cur_state)
        return list_state

    def run(self):
        for i in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % i)
            path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(i))
            if not os.path.exists(path_to_log):
                os.makedirs(path_to_log)  
            env = AnonEnv(path_to_log = path_to_log,
                            path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                            dic_traffic_env_conf = self.dic_traffic_env_conf)
            dict_state = env.reset()
            if self.model_name == "HRC" or self.model_name =="Single_PPO":
                self.agent.reset()
            reset_env_start_time = time.time()
            done = False
            step_num = 0
            reset_env_time = time.time() - reset_env_start_time
            running_start_time = time.time()
            while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
                step_start_time = time.time()
                # print("Multi_PPO_Neighbor rough", state)
                list_state = self.dict_to_list(dict_state)
                # print("Multi_PPO_Neighbor refine", state)

                if self.model_name == "Cenlight" or self.model_name == "Single_PPO" \
                        or self.model_name == "Bi_Cenlight" or self.model_name == "Double_Cenlight" \
                        or self.model_name == "RNN_Cenlight" or self.model_name == "HRC":
                    action = self.agent.choose_action(list_state)
                    list_action = self.agent.calculate_FT_offset(dict_state, action) if self.model_name == "HRC" or self.model_name == "Single_PPO" else action
                    print("agent action: ", action, " list action: ", list_action)
                    dict_next_state, multi_reward, done, _ = env.step(list_action)
                    print("time: {0}, running_time: {1}".format(env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                                time.time()-step_start_time))
                    store_reward = np.mean(multi_reward) if self.model_name == "HRC" or self.model_name == "Single_PPO" else multi_reward
                    self.agent.experience_store(list_state, action, store_reward)
                    if (step_num % 40 == 0 and step_num != 0) or done == True:
                        list_state = self.dict_to_list(dict_next_state)
                        self.agent.update(list_state)
                    dict_state = dict_next_state
                else:
                    if self.model_name != "Multi_PPO" or self.model_name != "Multi_PPO_Neighbor":
                        assert "\033[1;34m Undefined model name: {} \033[0m".format(self.model_name)

                    action = []
                    for agent_index in range(self.num_intersection):
                        action.append(self.agent[agent_index].choose_action(state[agent_index]))
                    action = np.array(action).reshape([-1])
                    next_state, reward, done, _ = env.step(action)
                    print("time: {0}, running_time: {1}".format(env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                                time.time()-step_start_time))
                    for agent_index in range(self.num_intersection):
                        self.agent[agent_index].experience_store(state[agent_index], action[agent_index], reward[agent_index])
                        if (step_num % 40 == 0 and step_num != 0) or done == True:
                            state = self.dict_to_list(next_state)
                            self.agent[agent_index].update(state[agent_index])
                    state = next_state

                step_num += 1
            running_time = time.time() - running_start_time

            log_start_time = time.time()
            print("start logging")
            env.bulk_log_multi_process()
            log_time = time.time() - log_start_time

            env.end_sumo()
            print("reset_env_time: ", reset_env_time)
            print("running_time: ", running_time)
            print("log_time: ", log_time)
