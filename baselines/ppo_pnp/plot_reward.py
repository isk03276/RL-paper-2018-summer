import matplotlib.pyplot as plt
import numpy as np

f1 = open("ppo_gail_pnp.txt", 'r')
f4 = open("ppo_interactive_gail_pnp.txt", 'r')
f5 = open("ppo_gail_cur_pnp.txt", 'r')
f6 = open("ppo_interactive_gail_cur_pnp.txt", 'r')

dppo1w_list = []
dppo4w_list = []
dppo5w_list = []
dppo6w_list = []
def read_reward(reward_list, f):
   while True:
      line = f.readline()
      if not line: break
      reward_list.append(float(line))

def get_dr_list(reward_list):
   t = 0
   dr_list = [reward_list[0]]
   for reward in reward_list: 
      if t > 16000:
          pass
      dr_list.append(reward)
      t += 1
   return dr_list

read_reward(dppo1w_list, f1)
read_reward(dppo4w_list, f4)
read_reward(dppo5w_list, f5)
read_reward(dppo6w_list, f6)

plt.plot(get_dr_list(dppo1w_list), label='RL + GAIL(PPO demo)')
plt.plot(get_dr_list(dppo4w_list), label='RL + interactive GAIL(PPO demo)')
plt.plot(get_dr_list(dppo5w_list), label='RL + GAIL + Curriculum(PPO demo)')
plt.plot(get_dr_list(dppo6w_list), label='RL + interactive GAIL + Curriculum(PPO demo)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Pick and Place')
#plt.legend(loc='lower right')
plt.legend()
plt.show()

f1.close()
f4.close()
