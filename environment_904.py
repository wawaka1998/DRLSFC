
#这个版本完成了_state中对node_last的添加1
#优化了部分代码（delay处改为 -delay而不是）
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import copy

import tensorflow as tf
import networkx as nx
import numpy as np
import tf_agents.metrics.tf_metrics

import sfcsim

from datetime import datetime

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network, q_network, sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import py_policy, py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
import shelve
from util import outputexcel

fail_reward = 0
success_reward = 1
scheduler_log = False
max_network_bw = 10.0
max_network_delay = 2.0
max_network_cpu = 10.0
max_nf_bw = 0.5*1.5*5  # max bw*ratio*num
max_nf_cpu = 3.75*2     # max nf_bw*rec_coef
max_nf_delay = 10.0
wait_time = 50
EXCEL_COL_OF_REWARD = "H"
EXCEL_COL_OF_DEPLOYED_NUMBER = "I"
DATE_OF_EXPERIMENT = "8.20"

network_file = shelve.open("./network_file/network")
network = network_file["cernnet2_1"]
network_file.close()



class NFVEnv(py_environment.PyEnvironment):

    def __init__(self, num_sfc=100):
        super().__init__()
        self._sfc_fin = False #部署完成一条sfc
        self._dep_fin = False
        self._dep_percent = 0.0
        self._num_sfc = num_sfc
        self.network = copy.deepcopy(network)
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = 0
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]     # processing sfc
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()       # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]        # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()      # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()        # remaining delay of sfc
        self._sfc_num_deployed = 0
        self._time = 0
        self._expiration_table = {}#过期时间表，记录着每个每个时间点下会过期的sfc(等待超时)，一旦过期，就会被放弃部署
        self._node_last = self.network.get_node(self._sfc_in_node)
        self._node_proc = None

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._node_num-1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._node_num * self._node_num + 3*self._node_num + 4, ), dtype=np.float32, minimum=0.0, name='observation'
        )
        self._generate_state()          #更新状态

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self, full_reset=False):
        # 全部sfc部署完成 清空网络开始下一组，重置一下状态
        self._dep_fin = False#代表所有sfc都已经部署完毕
        self._dep_percent = self._sfc_num_deployed / self.network.sfcs.get_number()
        # self.scheduler.show()
        self._time = 0
        self.network = copy.deepcopy(network)#这样的话就是每次都是同一个网络下部署同一批sfc
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = -1
        self._next_sfc()
        self._expiration_table = {}
        self._sfc_num_deployed = 0
        self._generate_state()#更新状态
        return ts.restart(self._state)

    def _step(self, action):
        self._time += 1
        #清理一下旧的sfc
        #这一块儿看看咋改，记得下一条sfc那里要提前ts.转换状态
        self._remove_sfc_run_out()
        if self._sfc_fin:
            self._sfc_fin = True
        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])#将要部署的node
        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc, weight='delay') #取两个节点间最短路径部署链路(延迟最小)
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        self._sfc_delay -= delay
        # 本次_step内,会把这个node部署完成
        if (self._sfc_delay < 0.0):
            return self._failed_in_this_step()


        if(self._deploy_node() == False or self._deploy_link(self._sfc_proc,self._vnf_index + 1,path) == False):
            return self._failed_in_this_step()

        # nf and link deploy success,but not the last one of this sfc
        if self._vnf_index < len(self._vnf_list) - 1:
            # not last vnf to deploy
            self._vnf_index += 1
            self._vnf_state_refresh()
            self._generate_state()#节点部署完毕，更新状态
            return ts.transition(self._state, reward=0.0)
        # last vnf, deploy the last link and end this episode

        if(self._deploy_final_link() == False):
            return self._failed_in_this_step()

        return self._deploy_this_sfc_successfully()


    def _remove_sfc_run_out(self):
        #去除那些已经部署超时的sfc
        expiration_times = self._expiration_table.keys()
        for expiration_time in list(expiration_times):
            if(expiration_time <= self._time):
                expiration_sfcs = self._expiration_table.pop(expiration_time)#弹出过期的那些sfcs
                for sfc in expiration_sfcs:
                    self.scheduler.remove_sfc(sfc, self.network)#

    def _generate_state(self):
        self.network_matrix.generate(self.network)
        b = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):#剩余带宽情况
            b = np.append(b, (self.network_matrix.get_edge_att('remain_bandwidth')[i][i+1:]) / max_network_bw)
        d = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):#延迟情况
            d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i+1:]) / max_network_delay)
        rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu #剩余节点情况
        in_node = np.zeros(self._node_num, dtype=np.float32)
        in_node[self.network_matrix.get_node_list().index(self._sfc_in_node)] = 1.0
        out_node = np.zeros(self._node_num, dtype=np.float32)
        out_node[self.network_matrix.get_node_list().index(self._sfc_out_node)] = 1.0
        last_node = np.zeros(self._node_num, dtype=np.float32)#上一个节点部署的位置
        if self._node_last is not None:
            last_node[self.network_matrix.get_node_list().index(self._node_last.id)] = 1.0
        self._state = np.concatenate((b, d, rsc, np.array([self._sfc_bw[self._vnf_index]/max_nf_bw], dtype=np.float32),#当前部署的需要的带宽
                                      np.array([self._vnf_detail[self._vnf_proc]['cpu']/max_nf_cpu], dtype=np.float32),#当前vnf所需要的cpu资源
                                      np.array([self._sfc_delay/max_nf_delay], dtype=np.float32),#当前sfc所剩余的延迟
                                      np.array([1.0], dtype=np.float32),
                                      in_node,
                                      out_node,
                                      last_node
                                      ), dtype=np.float32)
    def _wait_once(self):
        #这个函数需要搭配外层的while(true)使用
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time  :
            # 规定时间内部署不好就去掉整个sfc
            self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_fin = True
        else:
            # 等待，尝试下个回合接着部署
            self._time += 1
            self._remove_sfc_run_out()

    def _next_sfc(self):
        # 部署下一个sfc,包括刷新状态以及将时间推进到部署时刻
        self._sfc_index += 1
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        start_time = self._sfc_proc.get_atts()['start_time']
        self._time_out = False
        # 超过等待时间的不部署
        while not self._time <= start_time + wait_time:
            if self._sfc_index == self.network.sfcs.get_number() - 1:
                return self.reset()
            self._sfc_index += 1  # 可以跟上一句换换位置，然后就能去掉下面的try except了
            try:
                self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
            except:
                print(self._sfc_index)
            start_time = self._sfc_proc.get_atts()['start_time']

        # 没到等待时间的再等一下
        while not start_time <= self._time:
            self._remove_sfc_run_out()
            self._time += 1
        self._sfc_state_refresh()

    def _sfc_state_refresh(self):
        #对sfc状态进行刷新
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc
        self._node_last = self.network.get_node(self._sfc_in_node)
        self._node_proc = None
    def _vnf_state_refresh(self):
        #对vnf状态进行刷新
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self._node_last = self._node_proc  # 更新last_node

    def _print_deploy_result(self):
        print('Deployed {} / {}, clearing'.format(self._sfc_num_deployed, self.network.sfcs.get_number()))


    def _deploy_node(self):
        #部署节点,允许等待并再次尝试
        while(True):
            if not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc, self._vnf_index + 1, self._sfc_proc.get_vnf_types()) :
                # nf deploy failed
                self._wait_once()
                if self._sfc_fin:
                    return False
            else:
                return True

    def _deploy_link(self,sfc,node_number,path):
        #部署链接，允许等待并再次尝试
        while(True):
            if not self.scheduler.deploy_link(sfc, node_number, self.network, path):
                # link deploy failed,remove sfc
                self._wait_once()
                if self._sfc_fin:
                    return False
            else:
                break
        return True

    def _failed_in_this_step(self):
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            return self._deploy_this_sfc_failed(fail_reward)
        else:
            self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_state_refresh()  # 重新开始部署这条sfc
            self._generate_state()
            return ts.transition(self._state, reward=0)

    def _deploy_final_link(self):
        self._node_last = self._node_proc
        self._node_proc = self.network.get_node(self._sfc_out_node)
        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                weight='delay')
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc,
                                        weight='delay')
        self._sfc_delay -= delay
        if self._sfc_delay<0.0 or self._deploy_link(self._sfc_proc, self._vnf_index+2, path) == False:
            # link deploy failed
            return False
        return True

    def _deploy_this_sfc_failed(self,reward):
        self.scheduler.remove_sfc(self._sfc_proc, self.network)
        self._sfc_fin = True
        is_final =  self._sfc_index == (self.network.sfcs.get_number() - 1)
        if(is_final):
            self._dep_fin = True
            self._print_deploy_result()
            return ts.termination(self._state, reward)
        else:
            try:
                self._next_sfc()
                self._generate_state()
                return ts.transition(self._state, reward)
            except:
                raise Exception("发生错误,当前的sfc序号为" + str(self._sfc_index))


    def _deploy_this_sfc_successfully(self):
        is_final = (self._sfc_index == (self.network.sfcs.get_number() - 1))
        #  sfc deploy success
        self._sfc_num_deployed += 1
        expiration_time = self._time + self._sfc_proc.get_atts()['duration']
        if not expiration_time in self._expiration_table:
            self._expiration_table[expiration_time] = []
        self._expiration_table[expiration_time].append(self._sfc_proc)
        if (is_final):
            self._dep_fin = True
            self._print_deploy_result()
            return ts.termination(self._state, reward=self._sfc_proc.get_atts()['profit'])
        else:
            self._next_sfc()
            self._generate_state()
            return ts.transition(self._state, reward=self._sfc_proc.get_atts()['profit'])


    def get_info(self):
        return {
            'sfc_num_deployed': self._sfc_num_deployed
        }


if __name__ == '__main__':

    # environment = NFVEnv()
    # utils.validate_py_environment(environment, episodes=5)
    num_deployed = 100  # @param {type:"integer"}
    num_sfc = 1000  #代表要部署多少条sfc
    num_episodes = 40

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 5000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    shuffle = 32
    learning_rate = 0.0005  # @param {type:"number"}
    max_epsilon = 0.1 #包含
    min_epsilon = 0.1 #不包含
    target_update_tau = 0.95 #
    target_update_period = 500
    discount_gamma = 0.9

    num_parallel_calls = 8
    num_prefetch = batch_size

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    log_interval = 1  # @param {type:"integer"}

    checkpoint_dir = os.path.join('checkpoint/'+datetime.now().strftime("%Y%m%d-%H%M%S"), 'checkpoint')
    policy_dir = os.path.join('models/'+datetime.now().strftime("%Y%m%d-%H%M%S"), 'policy')
    log_dir = os.path.join('data/log', datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_py_env = NFVEnv(num_sfc)
    eval_py_env = NFVEnv(num_sfc)
    init_py_env = NFVEnv(num_sfc)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    init_env = tf_py_environment.TFPyEnvironment(init_py_env)

    fc_layer_params = (512, 256, 128)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    #activation存在于每一个神经元中，输入权值叠加后再通过激活函数输出
    #kernel_initializer指的是每一层的权重的初始化方法，这里是随机均匀分布
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.05, maxval=0.05, seed=None))


    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    # y = wx + b w指的是kernel，b是bias
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation= None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    #adam优化器也是梯度下降的
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    #train_step_counter = tf.Variable(0)
    #

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=target_update_period,
        gamma=discount_gamma,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()

    # replay buffer

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length= replay_buffer_max_length)

    # Add an observer that adds to the replay buffer:
    replay_observer = [replay_buffer.add_batch]
    random_policy = random_tf_policy.RandomTFPolicy(init_env.time_step_spec(), init_env.action_spec())

    initial_collect_op = dynamic_step_driver.DynamicStepDriver(
        init_env,
        random_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration
    )
    # initial collect data
    time_step = init_env.reset()
    step = 0
    while step < 10000 or not time_step.is_last():
         time_step = init_env.reset()
         while not time_step.is_last():
            step += 1
            time_step, _ = initial_collect_op.run(time_step)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=num_parallel_calls,
        sample_batch_size=batch_size,
        num_steps=2
    ).shuffle(shuffle).prefetch(num_prefetch)
    iterator = iter(dataset)#干什么用了？
    # train driver

    def update_driver(deploy_percent):
        epsilon = max_epsilon - (max_epsilon - min_epsilon) * deploy_percent
        train_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            epsilon_greedy_policy.EpsilonGreedyPolicy(agent.policy, epsilon = epsilon),
            observers=replay_observer,
            num_steps=collect_steps_per_iteration
        )
        return train_driver

    def output(num_deployed,total_reward):
        # output to excel
        outputexcel(DATE_OF_EXPERIMENT, EXCEL_COL_OF_DEPLOYED_NUMBER,str(episode + 3),num_deployed)
        outputexcel(DATE_OF_EXPERIMENT, EXCEL_COL_OF_REWARD, str(episode + 3), total_reward)
        print('Episode {} ,episode total reward: {}'.format(episode,total_reward))

    # main training loop
    train_driver = update_driver(0.0)
    for episode in range(num_episodes):
        if ((episode + 1) % (num_episodes//10) == 0):
            train_driver = update_driver(deploy_percent = (episode + 1) / num_episodes)

        total_reward = 0
        train_env.reset()
        time_step = train_env.current_time_step()
        #部署所有sfc
        while not time_step.is_last():
            time_step, _ = train_driver.run(time_step)
            # Sample a batch of data from the buffer and update the agent's network.
            trajectories, _ = next(iterator)
            agent.train(trajectories)
            total_reward += time_step.reward.numpy()[0]


        # save this episode's data
        #用打印以及写入excel
        if episode % log_interval == 0:
            num_deployed = train_env.pyenv.get_info()["sfc_num_deployed"][0]
            output(num_deployed,total_reward)





