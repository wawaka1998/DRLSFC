#这个版本完成了_state中对node_last的添加
#优化了部分代码（delay处改为 -delay而不是）

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import copy

import tensorflow as tf
import tensorflow_probability as tfp
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
from tf_agents.policies import py_policy
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


fail_reward = -3
success_reward = 1
scheduler_log = False
max_network_bw = 10.0
max_network_delay = 2.0
max_network_cpu = 10.0
max_nf_bw = 0.5*1.5*5  # max bw*ratio*num
max_nf_cpu = 3.75*2     # max nf_bw*rec_coef
max_nf_delay = 10.0
wait_time = 50
network_file = shelve.open("./network_file/network")
network = network_file["cernnet2_1"]
network_file.close()

class NFVEnv(py_environment.PyEnvironment):

    def __init__(self, num_sfc=100):
        super().__init__()
        self._episode_ended = False #部署完成一条sfc
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
        self._sfc_deployed = 0
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

        self.network_matrix.generate(self.network)

        self._generate_state()          #更新状态

    def _next_sfc(self):
        # 部署下一个sfc
        self._sfc_index += 1
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        start_time = self._sfc_proc.get_atts()['start_time']
        self._time_out = False
        # 超过等待时间的不部署
        while not self._time <= start_time + wait_time:
            if self._sfc_index == self.network.sfcs.get_number():
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
        self._episode_ended = False
        self._generate_state()


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self, full_reset=False):
        # 全部sfc部署完成 清空网络开始下一组，重置一下状态
        print('Deployed {} / {}, clearing'.format(self._sfc_deployed, self.network.sfcs.get_number()))
        self._dep_fin = False#代表所有sfc都已经部署完毕
        self._dep_percent = self._sfc_deployed / self.network.sfcs.get_number()
        # self.scheduler.show()
        self._time = 0
        self.network = copy.deepcopy(network)#这样的话就是每次都是同一个网络下部署同一批sfc
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = 0
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        self._sfc_deployed = 0
        self._sfc_state_refresh()
        self._expiration_table = {}
        self._episode_ended = False
        self._generate_state()#更新状态
        return ts.restart(self._state)

    def _step(self, action):
        self._time += 1
        #清理一下旧的sfc
        #这一块儿看看咋改，记得下一条sfc那里要提前ts.转换状态
        self._remove_sfc_run_out()
        if self._episode_ended:
            self._next_sfc()
        if self._dep_fin:
            self._end_and_reset()
        self.network_matrix.generate(self.network)#更新newtowrk_matrix矩阵
        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])#将要部署的node
        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc, weight='delay') #取两个节点间最短路径部署链路(延迟最小)
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        self._sfc_delay -= delay
        # 本次_step内,会把这个node部署完成
        if (self._sfc_delay < 0.0):
            self.scheduler.remove_sfc(self._sfc_proc, self.network)
            if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
                self._episode_ended = True
                self._generate_state()
                return ts.transition(self._state, reward = fail_reward)
            else:
                self._sfc_state_refresh()#重新开始部署这条sfc
                self._generate_state()
                return ts.transition(self._state, reward = 0)

        if(self._deploy_node() == False or self._deploy_link(self._sfc_proc,self._vnf_index + 1,path) == False):
            self._generate_state()
            return ts.transition(self._state, reward=fail_reward)

        #改到这里
        # nf link deploy success,but not the last one of this sfc
        if self._vnf_index < len(self._vnf_list) - 1:
            # not last vnf to deploy
            self._vnf_state_refresh()
            self._generate_state()#节点部署完毕，更新状态
            return ts.transition(self._state, reward=0.0)

        # last vnf, deploy the last link and end this episode
        else:
            return self._end_this_episode()

    def get_info(self):
        return {
            'dep_fin': self._dep_fin,
            'dep_percent': self._dep_percent
        }

    def _remove_sfc_run_out(self):
        #去除那些已经部署超时的sfc
        expiration_times = self._expiration_table.keys()
        for expiration_time in list(expiration_times):
            if(expiration_time <= self._time):
                expiration_sfcs = self._expiration_table.pop(expiration_time)#弹出过期的那些sfcs
                for sfc in expiration_sfcs:
                    self.scheduler.remove_sfc(sfc, self.network)#

    def _generate_state(self):
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
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            # 规定时间内部署不好就去掉整个sfc
            self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_index += 1
            self._episode_ended = True
        else:
            # 等待，尝试下个回合接着部署
            self._time += 1
            self._remove_sfc_run_out()
    def _sfc_state_refresh(self):
        #当需要部署下一条sfc时，对状态进行刷新
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc
        self.network_matrix.generate(self.network)
        self._node_last = self.network.get_node(self._sfc_in_node)
        self._node_proc = None
    def _vnf_state_refresh(self):
        self._vnf_index += 1
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self.network_matrix.generate(self.network)
        self._node_last = self._node_proc  # 更新last_node

    def _end_and_reset(self):
        print('Deployed {} / {}, clearing'.format(self._sfc_deployed, self.network.sfcs.get_number()))
        return self.reset()

    def _deploy_node(self):
        #部署节点,允许等待并再次尝试
        while(True):
            if not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc, self._vnf_index + 1, self._sfc_proc.get_vnf_types()) :
                # nf deploy failed
                self._wait_once()
                if self._episode_ended:
                    return False
            else:
                break

    def _deploy_link(self,sfc,node_number,path):
        #部署链接，允许等待并再次尝试
        while(True):
            if not self.scheduler.deploy_link(sfc, node_number, self.network, path):
                # link deploy failed,remove sfc
                self._wait_once()
                if self._episode_ended:
                    return False
            else:
                break
        return True

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

    def _end_this_episode(self):
        last_sfc = (self._sfc_index == self.network.sfcs.get_number())
        if not self._deploy_final_link():
            self.network_matrix.generate(self.network)
            self._node_last = None
            self._generate_state()
            self._sfc_index += 1
            self._sfc_deployed += 1
            self._episode_ended = True
            if (last_sfc):
                self._dep_fin = True
                self._generate_state()
                return ts.termination(self._state, reward= self._sfc_proc.get_atts()['profit'])
            else:
                self._generate_state()
                return ts.transition(self._state, reward= self._sfc_proc.get_atts()['profit'])

        #  sfc deploy success
        expiration_time = self._time + self._sfc_proc.get_atts()['duration']
        if not expiration_time in self._expiration_table:
            self._expiration_table[expiration_time] = []
        self._expiration_table[expiration_time].append(self._sfc_proc)

        if (last_sfc):
            self._dep_fin = True
            self._generate_state()
            return ts.termination(self._state, reward=self._sfc_proc.get_atts()['profit'])
        else:
            self._generate_state()
            return ts.transition(self._state, reward=self._sfc_proc.get_atts()['profit'])


if __name__ == '__main__':

    # environment = NFVEnv()
    # utils.validate_py_environment(environment, episodes=5)

    num_episodes = 100  # @param {type:"integer"}
    num_sfc = 1000  #代表要部署多少条sfc

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 5000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    shuffle = 32
    learning_rate = 0.0005  # @param {type:"number"}
    epsilon = 0.2#不按照最大价值函数更新的概率
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

    train_summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=10000)
    train_summary_writer.set_as_default()


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
        # target_update_tau=target_update_tau,
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
        step += 1
        time_step, _ = initial_collect_op.run(time_step)#最开始先用随机策略收集一些
    # print(replay_buffer.num_frames())
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=num_parallel_calls,
        sample_batch_size=batch_size,
        num_steps=2
    ).shuffle(shuffle).prefetch(num_prefetch)
    iterator = iter(dataset)#干什么用了？
    # train driver

    train_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(agent.policy, epsilon = epsilon)
    train_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        train_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration
    )

    train_checkpoint = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter
    )

    train_policy_saver = policy_saver.PolicySaver(agent.policy)

    train_checkpoint.initialize_or_restore()

    # main training loop
    for episode in range(num_episodes): #num_sfc = 100,num_episodes = 100 ; num_itr_per_episode = 200 搞不懂啥逻辑
        total_loss = 0
        total_reward = 0
        step_of_episode = 0
        num_deployed = 0
        while True:
            time_step = train_env.current_time_step()
            #部署所有sfc
            while not time_step.is_last():
                #直到terminal就结束这一轮
                # Collect a few steps and save to the replay buffer.
                time_step, _ = train_driver.run(time_step)
                # Sample a batch of data from the buffer and update the agent's network.
                trajectories, _ = next(iterator)
                train_loss = agent.train(trajectories).loss
                total_loss += train_loss
                total_reward += time_step.reward.numpy()[0]
                step_of_episode += 1

            train_env.reset()
            info = train_env.pyenv.get_info()
            if train_env.pyenv.get_info()['dep_fin'][0]:
                break

        # save this episode's data
        train_checkpoint.save(train_step_counter)

        if episode % log_interval == 0:
            print('Episode {}, step of this episode: {}, total step: {} ,episode total reward: {}, loss: {}'.format(episode, step_of_episode, agent.train_step_counter.numpy(),total_reward, total_loss / step))
            tf.summary.scalar('episode total reward', total_reward, step=episode)
            tf.summary.scalar('episode deployed percent', train_env.pyenv.get_info()['dep_percent'][0], step=episode)

    train_policy_saver.save(policy_dir)

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # compute_avg_return(eval_env, random_policy, num_eval_episodes)


