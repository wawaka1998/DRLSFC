import tensorflow as tf
from DEPLOY_ENV import DEPLOY_ENV
import shelve


import sys

modules = sys.modules
modules1 = sys.modules.copy()
from tf_agents.environments import tf_py_environment
modules2 = sys.modules.copy()
keys = modules2.keys() - modules1.keys()
modules1 = modules['tf_py_environment']
policy_dir = "./20221014-195340/policy"
policy = tf.saved_model.load(policy_dir)
network_file = shelve.open("./network_file/network")
network_and_sfc = network_file["cernnet2_6"]
network_file.close()
saved_policy = tf.saved_model.load(policy_dir)
eval_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)

