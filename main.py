from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DEPLOY_ENV import DEPLOY_ENV
import abc
import os
import copy

import tensorflow as tf
import networkx as nx
import numpy as np
import tf_agents.metrics.tf_metrics
from util import compute_avg_return
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
import heapq

FAIL_REWARD = -2 #这是整个sfc部署失败的惩罚,单步错误的惩罚会降低百分之10
scheduler_log = False
max_network_bw = 10.0
max_network_delay = 2.0
max_network_cpu = 10.0
max_nf_bw = 0.5 * 1.5 * 5  # max bw*ratio*num
max_nf_cpu = 3.75 * 2  # max nf_bw*rec_coef
max_nf_delay = 10.0
wait_time = 50
EXCEL_COL_OF_REWARD = "F"
EXCEL_COL_OF_DEPLOYED_NUMBER = "G"
DATE_OF_EXPERIMENT = "10.30"
CACULATE_TIME = 0.25
num_episodes = 110
MAX_epsilon = 0.9 # 包含
MID_epsilon = 0.15
MIN_epsilon = 0.02  # 不包含
discount_gamma = 0.9995



network_file = shelve.open("./network_file/network")
network_and_sfc = network_file["cernnet2_6"]
network_file.close()

if __name__ == '__main__':

    # environment = NFVEnv()
    # utils.validate_py_environment(environment, episodes=5)
    num_deployed = 100  # @param {type:"integer"}
    num_sfc = 1000  # 代表要部署多少条sfc


    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 300000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    shuffle = 32
    learning_rate = 0.0005  # @param {type:"number"}
    target_update_tau = 0.95  #
    target_update_period = 500


    num_parallel_calls = 8
    num_prefetch = batch_size

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    log_interval = 1  # @param {type:"integer"}

    checkpoint_dir = os.path.join('checkpoint/' + datetime.now().strftime("%Y%m%d-%H%M%S"), 'checkpoint')
    policy_dir = os.path.join('./' + datetime.now().strftime("%Y%m%d-%H%M%S"), 'policy')
    log_dir = os.path.join('data/log', datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_py_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)
    eval_py_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)
    init_py_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    init_env = tf_py_environment.TFPyEnvironment(init_py_env)

    fc_layer_params = (512, 256, 128)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    # activation存在于每一个神经元中，输入权值叠加后再通过激活函数输出
    # kernel_initializer指的是每一层的权重的初始化方法，这里是随机均匀分布
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.05, maxval=0.05, seed=None))


    def update_driver(epsilon):
        train_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            epsilon_greedy_policy.EpsilonGreedyPolicy(agent.policy, epsilon=epsilon),
            observers=replay_observer,
            num_steps=collect_steps_per_iteration
        )
        return train_driver


    def output(num_deployed, total_reward):
        # output to excel
        outputexcel(DATE_OF_EXPERIMENT, EXCEL_COL_OF_DEPLOYED_NUMBER, str(episode + 3), num_deployed,
                    datapath="./实验数据.xlsx")
        outputexcel(DATE_OF_EXPERIMENT, EXCEL_COL_OF_REWARD, str(episode + 3), total_reward, datapath="./实验数据.xlsx")
        print('Episode {} ,episode total reward: {}'.format(episode, total_reward))


    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    # y = wx + b w指的是kernel，b是bias
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    # adam优化器也是梯度下降的
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    # train_step_counter = tf.Variable(0)
    #

    def DEPLOY_ENV_action_constraint(observation):
        return observation['state'], observation['available_action']

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=target_update_period,
        gamma=discount_gamma,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        observation_and_action_constraint_splitter = DEPLOY_ENV_action_constraint
    )
    agent.initialize()

    # replay buffer

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size= train_env.batch_size,
        max_length=replay_buffer_max_length)

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
    #while step < 1000 or not time_step.is_last():
    while step < 100:
        time_step = init_env.reset()
        while step < 100:
        #while not time_step.is_last():
            step += 1
            time_step, _ = initial_collect_op.run(time_step)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=num_parallel_calls,
        sample_batch_size=batch_size,
        num_steps=2
    ).shuffle(shuffle).prefetch(num_prefetch)
    iterator = iter(dataset)  # 干什么用了？
    # train driver
    def caculate_epsilon_by_episodes(episode):
        if(episode < 71):
            deploy_percent = episode / 70
            epsilon = MAX_epsilon - (MAX_epsilon - MID_epsilon) * deploy_percent
        elif(episode < 91):
            deploy_percent = (episode - 70)/20
            epsilon = MID_epsilon - (MID_epsilon - MIN_epsilon) * deploy_percent
        else:
            epsilon = MIN_epsilon
        return epsilon

    # main training loop
    train_policy_saver = policy_saver.PolicySaver(agent.policy)
    train_driver = update_driver(epsilon = MAX_epsilon)
    for episode in range(1,num_episodes + 1):
        if ( episode  % (num_episodes // 40) == 0):
            epsilon = caculate_epsilon_by_episodes(episode)
            train_driver = update_driver(epsilon = epsilon)
        total_reward = 0
        train_env.reset()
        time_step = train_env.current_time_step()
        # 部署所有sfc
        while not time_step.is_last():
            time_step, _ = train_driver.run(time_step)
            # Sample a batch of data from the buffer and update the agent's network.
            total_reward += time_step.reward.numpy()[0]
            trajectories, _ = next(iterator)
            agent.train(trajectories)

        # save this episode's data
        # 用打印以及写入excel
        if episode % log_interval == 0:
            num_deployed = train_env.pyenv.get_info()["sfc_num_deployed"][0]
            output(num_deployed, total_reward)

    train_policy_saver.save(policy_dir)