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
network_file = shelve.open("./network_file/network")
network = network_file["cernnet2_3"]
network_file.close()
class NFVEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self._dep_fin = False
        self._dep_percent = 0.0
        self.network = network
        self._num_sfc = self.network.sfcs.number
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = 0
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc
        self._sfc_num_deployed = 0
        self._time = 0
        self._expiration_table = {}  # 过期时间表，记录着每个每个时间点下会过期的sfc(等待超时)，一旦过期，就会被放弃部署
        self._node_last = self.network.get_node(self._sfc_in_node)
        self._node_proc = None
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._node_num - 1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._node_num * self._node_num + 3 * self._node_num + 4,), dtype=np.float32, minimum=0.0,
            name='observation'
        )
        self._generate_state()  # 更新状态

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # 全部sfc部署完成 清空网络开始下一组，重置一下状态
        self._dep_fin = False  # 代表所有sfc都已经部署完毕
        self._dep_percent = self._sfc_num_deployed / self.network.sfcs.get_number()
        # self.scheduler.show()
        self._time = 0
        self.network = network  # 这样的话就是每次都是同一个网络下部署同一批sfc
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = -1
        self._next_sfc_waiting()
        self._expiration_table = {}
        self._sfc_num_deployed = 0
        self._generate_state()  # 更新状态
        return ts.restart(self._state)

    def _step(self, action):
        self._time += CACULATE_TIME
        while self._is_idle():
            self._remove_sfc_run_out()
            self._time += 1
            if not self._is_idle():
                self._generate_state()
                return ts.transition(self._state, reward=0)
        # 清理一下旧的sfc
        # 这一块儿看看咋改，记得下一条sfc那里要提前ts.转换状态
        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])  # 将要部署的node
        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                weight='delay')  # 取两个节点间最短路径部署链路(延迟最小)
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        self._sfc_delay -= delay
        # 本次_step内,会把这个node部署完成
        if (self._sfc_delay < 0.0):
            return self._failed_in_this_step()

        if (self._deploy_node() == False or self._deploy_link(self._sfc_proc, self._vnf_index + 1, path) == False):
            return self._failed_in_this_step()

        # nf and link deploy success,but not the last one of this sfc
        if self._vnf_index < len(self._vnf_list) - 1:
            # not last vnf to deploy
            self._vnf_index += 1
            self._vnf_state_refresh()
            self._generate_state()  # 节点部署完毕，更新状态
            return ts.transition(self._state, reward=0.0)
        # last vnf, deploy the last link and end this episode

        if (self._deploy_final_link() == False):
            return self._failed_in_this_step()

        return self._deploy_this_sfc_successfully()

    def _remove_sfc_run_out(self):
        # 去除那些已经部署超时的sfc
        expiration_times = self._expiration_table.keys()
        for expiration_time in list(expiration_times):
            if (expiration_time <= self._time):
                expiration_sfcs = self._expiration_table.pop(expiration_time)  # 弹出过期的那些sfcs
                for sfc in expiration_sfcs:
                    self.scheduler.remove_sfc(sfc, self.network)  #

    def _generate_state(self):
        self.network_matrix.generate(self.network)
        b = np.array([], dtype=np.float32)
        for i in range(self._node_num - 1):  # 剩余带宽情况
            b = np.append(b, (self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:]) / max_network_bw)
        d = np.array([], dtype=np.float32)
        for i in range(self._node_num - 1):  # 延迟情况
            d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i + 1:]) / max_network_delay)
        rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu  # 剩余节点情况
        in_node = np.zeros(self._node_num, dtype=np.float32)
        in_node[self.network_matrix.get_node_list().index(self._sfc_in_node)] = 1.0
        out_node = np.zeros(self._node_num, dtype=np.float32)
        out_node[self.network_matrix.get_node_list().index(self._sfc_out_node)] = 1.0
        last_node = np.zeros(self._node_num, dtype=np.float32)  # 上一个节点部署的位置
        if self._node_last is not None:
            last_node[self.network_matrix.get_node_list().index(self._node_last.id)] = 1.0
        self._state = np.concatenate(
            (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),  # 当前部署的需要的带宽
             np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),  # 当前vnf所需要的cpu资源
             np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),  # 当前sfc所剩余的延迟
             np.array([1.0], dtype=np.float32),
             in_node,
             out_node,
             last_node
             ), dtype=np.float32)

    def _next_sfc(self):
        # 部署下一个sfc,包括刷新状态
        self._sfc_index += 1
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        self._sfc_state_refresh()
        # 超过等待时间的不部署

    def _next_sfc_waiting(self):
        self._next_sfc()
        while self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            if self._sfc_index == self.network.sfcs.get_number() - 1:
                return self.reset()
            self._next_sfc()  # 可以跟上一句换换位置，然后就能去掉下面的try except了

    def _is_idle(self):
        if self._time < self._sfc_proc.get_atts()['start_time']:
            return True
        else:
            return False

    def _sfc_state_refresh(self):
        # 对sfc状态进行刷新(根据最新的self._proc)
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
        # 对vnf状态进行刷新
        self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
        self._node_last = self._node_proc  # 更新last_node

    def _print_deploy_result(self):
        print('Deployed {} / {}, clearing'.format(self._sfc_num_deployed, self.network.sfcs.get_number()))

    def _deploy_node(self):
        # 部署节点,允许等待并再次尝试
        if not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc, self._vnf_index + 1,
                                                  self._sfc_proc.get_vnf_types()):
            # nf deploy failed
            return False
        else:
            return True

    def _deploy_link(self, sfc, node_number, path):
        # 部署链接，允许等待并再次尝试
        if not self.scheduler.deploy_link(sfc, node_number, self.network, path):
            return False
        else:
            return True

    def _failed_in_this_step(self):
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            return self._deploy_this_sfc_failed(FAIL_REWARD)
        else:
            self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_state_refresh()  # 重新开始部署这条sfc
            self._generate_state()
            return ts.transition(self._state, reward =FAIL_REWARD / 100)

    def _deploy_final_link(self):
        self._node_last = self._node_proc
        self._node_proc = self.network.get_node(self._sfc_out_node)
        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                weight='delay')
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc,
                                        weight='delay')
        self._sfc_delay -= delay
        if self._sfc_delay < 0.0 or self._deploy_link(self._sfc_proc, self._vnf_index + 2, path) == False:
            # link deploy failed
            return False
        return True

    def _deploy_this_sfc_failed(self, reward = FAIL_REWARD):
        self.scheduler.remove_sfc(self._sfc_proc, self.network)
        is_final = self._sfc_index == (self.network.sfcs.get_number() - 1)
        if (is_final):
            self._dep_fin = True
            self._print_deploy_result()
            return ts.termination(self._state, reward)
        else:
            try:
                self._next_sfc_waiting()
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
            self._next_sfc_waiting()
            self._generate_state()
            return ts.transition(self._state, reward=self._sfc_proc.get_atts()['profit'])

    def get_info(self):
        return {
            'sfc_num_deployed': self._sfc_num_deployed
        }





