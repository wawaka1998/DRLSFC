import tensorflow as tf
from util import compute_avg_return
from DEPLOY_ENV import DEPLOY_ENV
import shelve
from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment

policy_dir = "./20221014-195340/policy"
policy = tf.saved_model.load(policy_dir)
network_file = shelve.open("./network_file/network")
network_and_sfc = network_file["cernnet2_6"]
network_file.close()
saved_policy = tf.saved_model.load(policy_dir)
eval_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)
eval_env = tf_py_environment.TFPyEnvironment(eval_env)
compute_avg_return(eval_env,saved_policy)