from openpyxl import load_workbook
import tensorflow as  tf
from tf_agents.policies import epsilon_greedy_policy
EXCEL_PATH = "./实验数据.xlsx"

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        #time_step = convert_TimeStep(time_step)
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step,())
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        print(episode_return)

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def convert_TimeStep(time_step):
    pass


def outputexcel(date:str,
           col:str,
           row:str,
           val:int,
           datapath = EXCEL_PATH
           ):
    """

    :date:实验日期(大约)，用于生成表格名称
    :col:列，从表格里看应该输出到哪一列
    :row:行，跟随episode变化
    :val:值，需要输出的数据值
    """
    sheet_name = date + '实验数据'
    situation = col + row
    wb = load_workbook(datapath)
    ws1 = wb[sheet_name]
    ws1[situation].value = int(val)
    wb.save(datapath)
def print_num(num):
    print(num)




