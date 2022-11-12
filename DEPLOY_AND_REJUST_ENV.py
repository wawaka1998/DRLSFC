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
import logging
import sys


ALTERNATIVE_NODE_NUM = 5
FAIL_REWARD = -2
success_reward = 1
scheduler_log = False
max_network_bw = 100.0
max_network_delay = 2.0
max_network_cpu = 30.0
max_nf_bw = 0.5 * 1.5 * 5  # max bw*ratio*num
max_nf_cpu = 3.75 * 2  # max nf_bw*rec_coef
max_nf_delay = 10.0
wait_time = 50
MIN_CPU_RESOURCE = 10
CPU_WARNING = MIN_CPU_RESOURCE *0.3
BANDWIDTH = 20
BANDWIDTH_WARNING = BANDWIDTH * 0.1
CACULATE_TIME = 0.25
ACTION_CONSTRAIN_NUM = 5
REJUST = 1
DEPLOY = 0
deploy_policy_dir = "./policies/policy2/policy"


class DEPLOY_AND_REJUST_ENV(py_environment.PyEnvironment):

    def __init__(self,network_and_sfc  = None):
        super().__init__()
        self._dep_fin = False
        self._dep_percent = 0.0
        self._network_and_sfc = network_and_sfc
        self.network = copy.deepcopy(self._network_and_sfc)
        self._num_sfc = self.network.sfcs.number
        self.scheduler = sfcsim.deploy_and_rejust_scheduler(log=scheduler_log)
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
        self._deploy_agent = tf.saved_model.load(deploy_policy_dir)
        self._rejust_pool = []
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum = 0, maximum=self._node_num , name='action'
        )
        self._recent_deployed_node = []
        self._observation_spec = {
            'state':array_spec.BoundedArraySpec(
                shape=(self._node_num * self._node_num + 3 * self._node_num + 4,), dtype = np.float32, minimum=0.0,
                name='observation'),
            'available_action': array_spec.BoundedArraySpec(shape=(self._node_num + 1,), dtype = np.int32,
                                                            minimum=0, maximum=1, name='available_action')
        }
        a = 1


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
        self.network = copy.deepcopy(self._network_and_sfc)  # 这样的话就是每次都是同一个网络下部署同一批sfc
        self.scheduler = sfcsim.deploy_and_rejust_scheduler(log = scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = -1
        self._next_not_timeout_sfc()
        self._expiration_table = {}
        self._sfc_num_deployed = 0
        self._recent_deployed_node = []
        self._rejust_pool = []
        return ts.restart(observation = self._generate_observation(type = REJUST))

    def _step(self, action):
        #可增加，如果不在rejust_pool跳过
        if (action != self._node_num):
            self._rejust(action)
        time_step = ts.transition(observation = self._get_deploy_observation_as_tensor(), reward = tf.constant([10.]))
        action_step = self._deploy_agent.action(time_step,())
        return self._deploy_step(action_step.action)

    def _rejust(self,action):
        #重点,仿造time_step
        rejust_node = self.network.get_node(self.network_matrix.get_node_list()[action])
        while(action in self._rejust_pool):
            rejust_vnf,target_node = self._get_target_node(rejust_node)
            if(target_node == None): break
            self._adjust_resource(rejust_node,target_node,rejust_vnf)
            self._update_rejust_pool()

    def _update_rejust_pool(self):
        #用于确定哪些需要节点需要调整,空余cpu小于MIN_CPU_RESOURCE * 0.3
        rejust_pool = []
        remain_cpu_resources = self._get_remain_cpu_resouces()
        for i in range(self._node_num):
            if(remain_cpu_resources[i] < CPU_WARNING):
                rejust_pool.append(i)
        self._rejust_pool = rejust_pool
        return rejust_pool
    #得到资源充足的节点中最近的那一个作为目标节点
    def _get_target_node(self,source):
        def _get_delay_without_source(sfc_id,vnf_number):
            sfc = self.network.sfcs.get_sfc(sfc_id)
            deploy_nodes = self.scheduler.get_records()[sfc_id]['node'].copy()
            num_nfs = len(deploy_nodes)
            deploy_nodes[0] = self.network.get_node(sfc.get_atts()['in_node'])
            deploy_nodes[num_nfs + 1] = self.network.get_node(sfc.get_atts()['out_node'])
            total_delay = 0
            for node_index in deploy_nodes.keys():
                if(vnf_number == node_index - 1 or vnf_number == node_index or node_index == num_nfs + 1):continue
                delay = nx.shortest_path_length(self.network.G, source = deploy_nodes[node_index], target=deploy_nodes[node_index + 1], weight='delay')
                total_delay += delay
            return total_delay
        answer = None
        remain_cpu_resources = self._get_remain_cpu_resouces()
        max_vnf = self._get_the_biggest_vnf_of_node(source.id)
        if(max_vnf == ""):
            return max_vnf,answer
        cpu_demand = self.scheduler.get_node_occupy_records()[source.id][max_vnf]['cpu']
        sfc_id, vnf_number = max_vnf.split("_")
        adjust_sfc = self.network.sfcs.get_sfc(sfc_id)
        delay_sum_without_source = _get_delay_without_source(sfc_id,vnf_number)
        remain_delay = adjust_sfc.get_atts()['delay'] - delay_sum_without_source
        min_delay = remain_delay
        for i in range(self._node_num):
            target_node = self.network.get_node(self.network_matrix.get_node_list()[i])
            remain_cpu_resource = remain_cpu_resources[i]
            if (remain_cpu_resource < cpu_demand + CPU_WARNING): continue  # cpu资源不够，排除
            traffic_last = adjust_sfc.atts['bandwidths'][int(vnf_number) - 1]
            traffic_next = adjust_sfc.atts['bandwidths'][int(vnf_number)]
            last_node,next_node = self._get_last_and_next_node(sfc_id,vnf_number)
            if (self._bandwidth_is_sufficient(source = last_node, target = target_node, traffic = traffic_last * 2 + BANDWIDTH_WARNING) == False):#带宽不足，淘汰
                continue
            if (self._bandwidth_is_sufficient(source = target_node , target = next_node, traffic = traffic_next * 2+ BANDWIDTH_WARNING) == False):#带宽不足，淘汰
                continue
            delay = nx.shortest_path_length(self.network.G, source = last_node, target=target_node, weight='delay') \
                    + nx.shortest_path_length(self.network.G, source=target_node, target=next_node, weight='delay')
            if (delay < min_delay): answer = target_node  # 延迟太高，排除
        return max_vnf,answer
    def _adjust_resource(self,rejust_node,target_node,rejust_vnf):
        def delete_resource_and_record():
            sfc = self.network.sfcs.get_sfc(sfc_id)
            self.scheduler.remove_nf(sfc,int(vnf_number))
            self.scheduler.remove_link(sfc,int(vnf_number),self.network)
            self.scheduler.remove_link(sfc, int(vnf_number) + 1, self.network)
        def deploy_new_node():
            sfc = self.network.sfcs.get_sfc(sfc_id)
            #self.scheduler.deploy_nf(sfc,target_node,int(vnf_number))
            self.scheduler.deploy_nf_scale_out(sfc, target_node,int(vnf_number) ,sfc.get_vnf_types())
            last_node,next_node = self._get_last_and_next_node(sfc_id,vnf_number)
            self._deploy_link(sfc,int(vnf_number),last_node,target_node)
            self._deploy_link(sfc, int(vnf_number) + 1,target_node,next_node)

        sfc_id, vnf_number = rejust_vnf.split("_")
        self.scheduler.log = True
        delete_resource_and_record()
        deploy_new_node()
        self.scheduler.log = False
    def _get_the_biggest_vnf_of_node(self,node_id):
        node_occupy = self.scheduler.get_node_occupy_records()[node_id]
        max = 0
        max_vnf = ""
        keys = list(node_occupy.keys())
        for key in keys:
            sfc_id, vnf_number = key.split("_")
            sfc = self.network.sfcs.get_sfc(sfc_id)
            if(len(self.scheduler.get_records()[sfc_id]['node']) != len(sfc.get_nfs())):
                continue
            cpu = node_occupy[key]['cpu']
            if(cpu > max):
                max_vnf = key
                max = cpu
        return max_vnf
    def _deploy_step(self,action):
        action = action.numpy()
        action = action[0]
        self._time_passed(CACULATE_TIME)
        if(action == self._node_num):
            self._time_passed(1)
            return self._failed_in_this_step(no_available_action = True)
        while self._is_idle():
            #没有sfc请求到达
            self._time_passed(1)
            if not self._is_idle():
                return ts.transition(observation = self._generate_observation(type = REJUST), reward = 0)
        # 清理一下旧的sfc
        # 这一块儿看看咋改，记得下一条sfc那里要提前ts.转换状态

        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])  # 将要部署的node
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        # 本次_step内,会把这个node部署完成
        if (self._sfc_delay < delay):
            return self._failed_in_this_step()
        if (self._deploy_node() == False or self._deploy_link(self._sfc_proc, self._vnf_index + 1, self._node_last,self._node_proc) == False):
            return self._failed_in_this_step()
        if self._vnf_index < len(self._vnf_list) - 1:
            return self._deploy_this_vnf_successfully(delay)

        #if the last vnf,need to deploy last_link
        while(self._time <= self._sfc_proc.get_atts()['start_time'] + wait_time):
            if (self._deploy_final_link()):
                return self._deploy_this_sfc_successfully()
            else:
                self._time_passed(1)
        return self._deploy_this_sfc_failed()

    def _remove_sfc_run_out(self):
        # 去除那些已经部署超时的sfc
        expiration_times = self._expiration_table.keys()
        for expiration_time in list(expiration_times):
            if (expiration_time <= self._time):
                expiration_sfcs = self._expiration_table.pop(expiration_time)  # 弹出过期的那些sfcs
                for sfc in expiration_sfcs:
                    self.scheduler.remove_sfc(sfc, self.network)  #

    def _get_available_deploy_nodes(self, source, actions_num):
        available_nodes = []
        remain_cpu_resources = self._get_remain_cpu_resouces()
        cpu_demand = self._vnf_detail[self._vnf_proc]['cpu']
        for i in range(self._node_num):
            target_node = self.network.get_node(self.network_matrix.get_node_list()[i])
            cpu = remain_cpu_resources[i]
            if(i in self._recent_deployed_node):continue #最近部署过，排除
            if(cpu < cpu_demand): continue    #cpu资源不够，排除
            delay = nx.shortest_path_length(self.network.G, source= source ,target = target_node, weight='delay')
            traffic = self._sfc_proc.atts['bandwidths'][self._vnf_index]
            if(self._bandwidth_is_sufficient(source =  source,target = target_node,traffic = traffic) == False):
                continue
            if(self._vnf_index == len(self._vnf_list) - 1):
                traffic = self._sfc_proc.atts['bandwidths'][self._vnf_index + 1]
                if(self._bandwidth_is_sufficient(source = target_node,target=self.network.get_node(self._sfc_out_node),traffic= traffic) == False):
                    continue
                delay += nx.shortest_path_length(self.network.G, source = target_node,target = self.network.get_node(self._sfc_out_node), weight='delay')
            if (delay > self._sfc_delay): continue  # 延迟太高，排除
            available_node = [i,delay]
            available_nodes.append(available_node)
        available_nodes.sort(key = lambda x: x[1], reverse = False)#top 5 shortest
        available_actions = []
        for i in range(actions_num):
            if(i < len(available_nodes)):
                available_actions.append(available_nodes[i][0])
        if(len(available_actions) == 0):
            available_actions.append(self._node_num)#action = self._node_num 代表没有可选动作
        np_available_actions = np.zeros(self._node_num + 1,dtype = np.int32)
        for action in available_actions:
            np_available_actions[action] = 1
        return np_available_actions

    def _generate_observation(self,type = REJUST):
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
        last_node_index = self.network_matrix.get_node_list().index(self._node_last.id)
        last_node[last_node_index] = 1.0
        state = np.concatenate(
            (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),  # 当前部署的需要的带宽
             np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),  # 当前vnf所需要的cpu资源
             np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),  # 当前sfc所剩余的延迟
             np.array([1.0], dtype=np.float32),
             in_node,
             out_node,
             last_node
             ), dtype=np.float32)
        if(type == 1):
            self._update_rejust_pool()
            rejust_pool = np.zeros(self._node_num + 1,dtype = np.int32)
            for action in self._rejust_pool:
                rejust_pool[action] = 1
            if(len(self._rejust_pool) == 0):
                rejust_pool[self._node_num] = 1
            observation = {'state': state,'available_action':rejust_pool}
        if(type == 0):
            available_actions = self._get_available_deploy_nodes(source = self._node_last,actions_num = 5)
            observation = {'state': state, 'available_action': available_actions}
        return observation

    def _get_deploy_observation_as_tensor(self):
        observation = self._generate_observation(type = DEPLOY)
        observation['state'] = tf.convert_to_tensor(np.expand_dims(observation['state'], axis=0))
        observation['available_action'] = tf.convert_to_tensor(np.expand_dims(observation['available_action'], axis=0))
        return observation

    def _next_sfc(self):
        # 部署下一个sfc,包括刷新状态
        self._sfc_index += 1
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
        self._sfc_state_refresh()
        # 超过等待时间的不部署

    def _next_not_timeout_sfc(self):
        #deploy sfc not timeout
        self._next_sfc()
        while self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            if self._sfc_index == self.network.sfcs.get_number() - 1:
                return self.reset()
            self._next_sfc()  # 可以跟上一句换换位置，然后就能去掉下面的try except了

    def _is_idle(self):
        #没有部署任务
        if self._time < self._sfc_proc.get_atts()['start_time']:
            return True
        else:
            return False


    def _bandwidth_is_sufficient(self,source,target,traffic,path = None):
        if(path == None):
            path = nx.shortest_path(self.network.G, source = source, target = target,weight='delay')    # 取两个节点间最短路径部署链路(延迟最小)
        for i in range(len(path) - 1):
            bandwidth1 = self.network.G[path[i]][path[i + 1]]['remain_bandwidth']
            if bandwidth1 < traffic:
                return False
        return True

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

    def _deploy_link(self, sfc, node_number, source_node,target_node,path = None):
        # 部署链接，允许等待并再次尝试
        if(path == None):
            path = nx.shortest_path(self.network.G, source = source_node, target = target_node,
                                weight='delay')  # 取两个节点间最短路径部署链路(延迟最小)
        if not self.scheduler.deploy_link(sfc, node_number, self.network, path):
            return False
        else:
            return True

    def _failed_in_this_step(self, no_available_action = False):
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            if(no_available_action):
                return self._deploy_this_sfc_failed(0)
            else:
                return self._deploy_this_sfc_failed(FAIL_REWARD)
        else:
            #self.scheduler.remove_sfc(self._sfc_proc, self.network)
            #self._sfc_state_refresh()  # 重新开始部署这条sfc
            if(no_available_action):
                return ts.transition(observation=self._generate_observation(type = REJUST), reward = 0)
            else:
                return ts.transition(observation = self._generate_observation(type = REJUST), reward =FAIL_REWARD / 50)

    def _deploy_final_link(self):
        source = self._node_proc
        target = self.network.get_node(self._sfc_out_node)
        delay = nx.shortest_path_length(self.network.G, source = source, target = target,
                                        weight='delay')
        if self._sfc_delay < delay or self._deploy_link(self._sfc_proc, self._vnf_index + 2,
                                                        source_node = source,target_node = target) == False:
            # link deploy failed
            return False
        self._sfc_delay -= delay
        return True

    def _deploy_this_sfc_failed(self, reward = FAIL_REWARD):
        self.scheduler.remove_sfc(self._sfc_proc, self.network)
        is_final = self._sfc_index == (self.network.sfcs.get_number() - 1)
        if (is_final):
            self._dep_fin = True
            self._print_deploy_result()
            return ts.termination(observation = self._generate_observation(type = REJUST),reward = reward)
        else:
            try:
                self._next_not_timeout_sfc()
                return ts.transition(observation = self._generate_observation(type = REJUST), reward = reward)
            except:
                raise Exception("发生错误,当前的sfc序号为" + str(self._sfc_index))

    def _deploy_this_vnf_successfully(self,delay):
        # nf and link deploy success,but not the last one of this sfc
        self._sfc_delay -= delay
        self._vnf_index += 1
        self._vnf_state_refresh()
        return ts.transition(observation = self._generate_observation(type = REJUST), reward=0.0)


    def _deploy_this_sfc_successfully(self):
        is_final = (self._sfc_index == (self.network.sfcs.get_number() - 1))
        #  sfc deploy success
        self._sfc_num_deployed += 1
        print("sfc" + str(self._sfc_index + 1) + "已部署")
        expiration_time = self._time + self._sfc_proc.get_atts()['duration']
        if not expiration_time in self._expiration_table:
            self._expiration_table[expiration_time] = []
        self._expiration_table[expiration_time].append(self._sfc_proc)
        if (is_final):
            self._dep_fin = True
            self._print_deploy_result()
            return ts.termination(observation = self._generate_observation(type = REJUST), reward=self._sfc_proc.get_atts()['profit'])
        else:
            self._next_not_timeout_sfc()
            return ts.transition(observation = self._generate_observation(type = REJUST), reward=self._sfc_proc.get_atts()['profit'])

    def _time_passed(self,duration):
        self._time += duration
        self._remove_sfc_run_out()

    def _add_recent_deploy(self,action):
        #将最近几次部署的位置记录下来，如果超过5个就清空
        if(len(self._recent_deployed_node) >= ALTERNATIVE_NODE_NUM):
            self._recent_deployed_node = []
        self._recent_deployed_node.append(action)

    def _get_last_and_next_node(self,sfc_id,vnf_number):
        sfc = self.network.sfcs.get_sfc(sfc_id)
        if (int(vnf_number) > 1):
            last_node = self.scheduler.get_records()[sfc_id]['node'][int(vnf_number) - 1]
        else:
            last_node = self.network.get_node(sfc.get_in_node())
        if (int(vnf_number) < len(sfc.get_atts()['nfs'])):
            next_node = self.scheduler.get_records()[sfc_id]['node'][int(vnf_number) + 1]
        else:
            next_node = self.network.get_node(sfc.get_out_node())
        return last_node,next_node

    def _get_remain_cpu_resouces(self):
        self.network_matrix.generate(self.network)
        remain_cpu_resources = self.network_matrix.get_node_atts('cpu')
        return remain_cpu_resources
    def get_info(self):
        return {
            'sfc_num_deployed': self._sfc_num_deployed
        }





