from util import compute_avg_return
from DEPLOY_ENV import DEPLOY_ENV
from tf_agents.environments import tf_py_environment
import tensorflow as tf
import shelve
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.networks import network, q_network, sequential
from tf_agents.utils import common

network_file = shelve.open("./network_file/network")
network_and_sfc = network_file["cernnet2_7"]
network_file.close()
eval_py_env = DEPLOY_ENV(network_and_sfc = network_and_sfc)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
policy_dir = "./policies/policy1/policy"
policy = tf.saved_model.load(policy_dir)


discount_gamma = 0.9995
learning_rate = 0.0005
target_update_tau = 0.95  #
target_update_period = 500
fc_layer_params = (512, 256, 128)
action_tensor_spec = tensor_spec.from_spec(eval_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.05, maxval=0.05, seed=None))

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
    eval_env.time_step_spec(),
    eval_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=target_update_period,
    gamma=discount_gamma,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=DEPLOY_ENV_action_constraint
)
#测试已训练policy
#avg_return = compute_avg_return(eval_env,policy)
agent.policy = policy
agent.initialize()

eval_driver = dynamic_step_driver.DynamicStepDriver(
    eval_env,
    epsilon_greedy_policy.EpsilonGreedyPolicy(agent.policy, epsilon = 0)
)

#测试未训练智能体
for episode in range(1, 11):
    total_reward = 0
    eval_env.reset()
    time_step = eval_env.current_time_step()
    # 部署所有sfc
    while not time_step.is_last():
        time_step, _ = eval_driver.run(time_step)
        # Sample a batch of data from the buffer and update the agent's network.
        total_reward += time_step.reward.numpy()[0]


