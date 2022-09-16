from openpyxl import load_workbook


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def output(date:str,
           col:str,
           row:str,
           val:int,
           datapath = './实验数据.xlsx'
           ):
    sheet_name = date + '实验数据'
    situation = col + row
    wb = load_workbook(datapath)
    ws1 = wb[sheet_name]
    ws1[situation].value = val
    cellB2_value = ws1['A2'].value
    wb.save("./实验数据.xlsx")
