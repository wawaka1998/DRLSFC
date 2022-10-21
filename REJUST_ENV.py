# 这个版本完成了_state中对node_last的添加1
# 优化了部分代码（delay处改为 -delay而不是）
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
import heapq
from DEPLOY_ENV import DEPLOY_ENV
FAIL_REWARD = -2
success_reward = 1
scheduler_log = False
max_network_bw = 10.0
max_network_delay = 2.0
max_network_cpu = 10.0
max_nf_bw = 0.5 * 1.5 * 5  # max bw*ratio*num
max_nf_cpu = 3.75 * 2  # max nf_bw*rec_coef
max_nf_delay = 10.0
wait_time = 50
EXCEL_COL_OF_REWARD = "B"
EXCEL_COL_OF_DEPLOYED_NUMBER = "C"
DATE_OF_EXPERIMENT = "9.23"
CACULATE_TIME = 0.25


class REJUST_ENV(DEPLOY_ENV):
    def __init__(self,network):
        super().__init__(network)
        self._adjust_queue = self._init_adjust_queue()

    def _reset(self):
        pass

    def _step(self, action):
        pass

    def _init_adjust_queue(self):
        node_list = []
        for node in self.network.nodes:
            node_list.append(node)
        node_list.sort(key=lambda x: x.atts["cpu"] / x.remain_resource["cpu"], reverse=True)
        return node_list
    def _refresh_adjust_queue(self):
        self._adjust_queue.sort(key=lambda x: x.atts["cpu"] / x.remain_resource["cpu"], reverse=True)