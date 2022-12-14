from sfcsim.classes.network import *
from sfcsim.classes.sfc import *
from sfcsim.layout.cernnet2_layout import *
import random
from random import choice



def calculate_profit(bandwidth, vnf_of_sfc, duration, vnf_types):
    total_profit = 0
    for vnf in vnf_of_sfc:
        total_profit += bandwidth * vnf_types.get_vnf_type(vnf).get_coeff()['cpu'] * duration
    return total_profit / 200


def make_start_times_list(num_sfc):
    start_times_list = []
    for i in range(num_sfc):
        start_time = int(random.uniform(1200, 10000))
        start_times_list.append(start_time)
    return sorted(start_times_list)

class cernnet2_sfc_dynamic(network):
    def __init__(self, num_sfc = 100, network_name = "cernnet2"):
        self.name = network_name
        self.node1=node(uuid='node1',atts={'cpu':10,'access':False})
        self.node2=node(uuid='node2',atts={'cpu':10,'access':False})
        self.node3=node(uuid='node3',atts={'cpu':10,'access':False})
        self.node4=node(uuid='node4',atts={'cpu':10,'access':False})
        self.node5=node(uuid='node5',atts={'cpu':10,'access':False})
        self.node6=node(uuid='node6',atts={'cpu':10,'access':False})
        self.node7=node(uuid='node7',atts={'cpu':10,'access':False})
        self.node8=node(uuid='node8',atts={'cpu':10,'access':False})
        self.node9=node(uuid='node9',atts={'cpu':10,'access':False})
        self.node10=node(uuid='node10',atts={'cpu':10,'access':False})
        self.node11=node(uuid='node11',atts={'cpu':10,'access':False})
        self.node12=node(uuid='node12',atts={'cpu':10,'access':False})
        self.node13=node(uuid='node13',atts={'cpu':10,'access':False})
        self.node14=node(uuid='node14',atts={'cpu':10,'access':False})
        self.node15=node(uuid='node15',atts={'cpu':10,'access':False})
        self.node16=node(uuid='node16',atts={'cpu':10,'access':False})
        self.node17=node(uuid='node17',atts={'cpu':10,'access':False})
        self.node18=node(uuid='node18',atts={'cpu':10,'access':False})
        self.node19=node(uuid='node19',atts={'cpu':10,'access':False})
        self.node20=node(uuid='node20',atts={'cpu':10,'access':False})
        self.node21=node(uuid='node21',atts={'cpu':10,'access':False})
        server_nodes=[self.node1,self.node2,self.node3,self.node4,self.node5,self.node6,self.node7,self.node8,self.node9,self.node10,\
                   self.node11,self.node12,self.node13,self.node14,self.node15,self.node16,self.node17,self.node18,self.node19,self.node20,self.node21]
        access_nodes=[]
        network.__init__(self,server_nodes+access_nodes)
        self.generate_edges()
        self.generate_nodes_atts()
        self.generate_edges_atts()
        self.vnf_types=vnf_types(vnf_types=[(vnf_type(name='type1',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1}))\
                        ,vnf_type(name='type2',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1})\
                        ,vnf_type(name='type3',atts={'cpu':0},ratio=1.2,resource_coefficient={'cpu':1.8})\
                        ,vnf_type(name='type4',atts={'cpu':0},ratio=1.5,resource_coefficient={'cpu':1.5})\
                        ,vnf_type(name='type5',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':1.4})\
                        ,vnf_type(name='type6',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':1.2})\
                        ,vnf_type(name='type7',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1.2})\
                        ,vnf_type(name='type8',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':2})])
        vnf_list = ['type1','type2','type3','type4','type5','type6','type7','type8']
        sfc_list = []

        start_times_list = make_start_times_list(num_sfc)
        for i in range(1, num_sfc+1):
            length = random.randint(3,5)
            vnf_of_sfc = random.sample(vnf_list,length)
            bandwidth_required = round(random.uniform(0.48,0.52),2)
            delay_constraint = round(random.uniform(10.0,18.0),2)
            duration = int(random.uniform(1000,1800))
            start_time = start_times_list[i - 1]
            profit = round(calculate_profit(bandwidth_required, vnf_of_sfc, duration, self.vnf_types),2)
            s = sfc(uuid = 'sfc'+str(i),in_node = choice(server_nodes).get_id(), out_node = choice(server_nodes).get_id(), nfs = vnf_of_sfc,
                    bandwidth = bandwidth_required, delay = delay_constraint, start_time= start_time,duration = duration,profit=profit,
                    vnf_types=self.vnf_types)
            sfc_list.append(s)
        self.sfcs=sfcs(sfc_list)
        self.figure=''

    def generate_edges(self):
        self.add_edges([[self.node1,self.node2,{'bandwidth':20}],[self.node2,self.node3,{'bandwidth':20}],\
                        [self.node3,self.node4,{'bandwidth':20}],[self.node3,self.node5,{'bandwidth':20}],\
                        [self.node5,self.node6,{'bandwidth':20}],[self.node5,self.node7,{'bandwidth':20}],\
                        [self.node5,self.node9,{'bandwidth':20}],[self.node5,self.node16,{'bandwidth':20}],\
                        [self.node6,self.node8,{'bandwidth':20}],[self.node7,self.node9,{'bandwidth':20}],\
                        [self.node8,self.node12,{'bandwidth':20}],[self.node9,self.node10,{'bandwidth':20}],\
                        [self.node10,self.node11,{'bandwidth':20}],[self.node12,self.node13,{'bandwidth':20}],\
                        [self.node12,self.node14,{'bandwidth':20}],[self.node13,self.node15,{'bandwidth':20}],\
                        [self.node14,self.node16,{'bandwidth':20}],[self.node15,self.node20,{'bandwidth':20}],\
                        [self.node16,self.node17,{'bandwidth':20}],[self.node16,self.node19,{'bandwidth':20}],\
                        [self.node16,self.node21,{'bandwidth':20}],[self.node17,self.node18,{'bandwidth':20}],[self.node20,self.node21,{'bandwidth':20}]])
    def generate_nodes_atts(self,atts=[30, 29, 28, 27, 27, 27, 26, 22, 22, 20, 19, 17, 16, 16, 14, 14, 13, 13, 12, 11, 10]):
        nodes=[5,16,21,3,12,13,10,1,2,4,6,7,8,9,11,14,15,17,18,19,20]
        if len(atts)==len(nodes):
            i=0
            for node in nodes:
                self.set_atts('node'+str(node),{'cpu':atts[i]})
                i+=1
    def generate_edges_atts(self,atts=[0.77, 0.59, 1.47, 0.95, 0.59, 0.69, 1.56, 1.1, 0.52, 1.03, 0.95, 1.08, 0.83, 1.21, 1.33, 0.92, 0.75, 1.34, 1.22, 1.29, 0.56, 0.64, 1.3]):
        i=0
        for edge in self.G.edges:
            self.set_edge_atts(edge[0],edge[1],{'delay':atts[i]})
            i+=1
    def draw(self,figsize=[36,20],node_size=10000,node_fone_size=8,link_fone_size=9,node_shape='H',path=''):
        network.draw(self,figsize=figsize,pos=cernnet2_layout(self.G),node_size=node_size,node_fone_size=node_fone_size,link_fone_size=link_fone_size,node_shape=node_shape)
    def draw_dynamic(self,figsize=[36,20],path='',node_size=10000,node_fone_size=8,link_fone_size=9,node_shape='H'):
        network.draw_dynamic(self,figsize=figsize,pos=cernnet2_layout(self.G),node_size=node_size,node_fone_size=node_fone_size,link_fone_size=link_fone_size,node_shape=node_shape)
