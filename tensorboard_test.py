while (True):  # 本次_step内,会把这个node部署完成
    if self._sfc_delay < 0.0 or not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc,
                                                                       self._vnf_index + 1,
                                                                       self._sfc_proc.get_vnf_types()):
        # nf deploy failed
        if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
            # 规定时间内部署不好就去掉
            if self._vnf_index != 0:
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_index += 1
            self._episode_ended = True
            return ts.termination(self._state, reward=fail_reward)
        else:
            # 等待，尝试下个回合接着部署
            self._time += 1
            self._remove_sfc_run_out()
    else:
        if not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index + 1, self.network, path):
            # link deploy failed
            # remove sfc
            if self._time > self._sfc_proc.get_atts()['start_time'] + wait_time:
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
                self._sfc_index += 1

                # ending this episode
                self._episode_ended = True
                return ts.termination(self._state, reward=fail_reward)
            else:
                self._time += 1
                self._remove_sfc_run_out()
        else:
            # nf link deploy success,but not the last one of this sfc
            if self._vnf_index < len(self._vnf_list) - 1:
                # not last vnf to deploy
                self._vnf_index += 1
                self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
                self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr

                self.network_matrix.generate(self.network)
                self._generate_state()  # 节点部署完毕，更新状态
                return ts.transition(self._state, reward=0.0)

            else:
                # last vnf, deploy the last link部署个最后一个link，也太臭长了
                self._node_last = self._node_proc
                self._node_proc = self.network.get_node(self._sfc_out_node)
                path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                        weight='delay')
                delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc,
                                                weight='delay')
                self._sfc_delay -= (delay / max_nf_delay)
                if self._sfc_delay < 0.0 or not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index + 2,
                                                                           self.network, path):
                    # link deploy failed
                    # remove sfc
                    self.scheduler.remove_sfc(self._sfc_proc, self.network)
                    self._sfc_index += 1
                    # ending this episode
                    self._episode_ended = True
                    return ts.termination(self._state, reward=fail_reward)
                else:
                    # sfc deploy success

                    self.network_matrix.generate(self.network)
                    self._generate_state()
                    self._sfc_index += 1
                    self._sfc_deployed += 1
                    # ending this episode
                    self._episode_ended = True
                    # reward = success_reward * (self._sfc_deployed / self.network.sfcs.get_number())
                    expiration_time = self._time + self._sfc_proc.get_atts()['duration']
                    if not expiration_time in self._expiration_table:
                        self._expiration_table[expiration_time] = []
                    self._expiration_table[expiration_time].append(self._sfc_proc)
                    return ts.termination(self._state, reward=self._sfc_proc.get_atts()['profit'])
