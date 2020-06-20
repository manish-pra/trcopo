# Game imports
import os
import sys
sys.path.insert(0, '../..')
from open_spiel.python import rl_environment
import torch
from others.maddpg.buffer import ReplayBuffer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from others.maddpg.utils.maddpg_c import MADDPG

from MarkovSoccer.soccer_state import get_two_state

folder_location = 'tensorboard/soccer/'
experiment_name = 'maddpg/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)

writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

maddpg1 = MADDPG(num_in_pol=12, num_out_pol=5, num_in_critic=24,lr = 0.005,discrete_action=True,agent_i=1)
maddpg2 = MADDPG(num_in_pol=12, num_out_pol=5, num_in_critic=24,lr = 0.005,discrete_action=True,agent_i=2)

batch_size = 10
sample_size = 400
num_episode = 10001
buffer_length = 600
game = "markov_soccer"
num_players = 2

env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]
print(num_actions)

replay_buffer = ReplayBuffer(buffer_length, num_players,
                             [12,12],
                             [5,5])


for t_eps in range(0,num_episode,1):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []

    #data_collection
    avg_itr = 0
    for _ in range(batch_size):
        i = 0
        time_step = env.reset()
        a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
        while time_step.last()==False:

            obs1 = torch.FloatTensor([rel_state1])
            obs2 = torch.FloatTensor([rel_state2])
            action1 = maddpg1.step(obs1, explore=True)
            action2 = maddpg2.step(obs2, explore=True)

            action = [action1.data.numpy(), action2.data.numpy()]

            env_action = [torch.argmax(action1),torch.argmax(action2)]

            time_step = env.step(env_action)
            a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
            next_obs1 = torch.tensor([rel_state1])
            next_obs2 = torch.tensor([rel_state2])


            reward1 = time_step.rewards[0]
            reward2 = time_step.rewards[1]

            #print(pretty_board(time_step))
            mat_reward1.append(torch.FloatTensor([reward1]))
            mat_reward2.append(torch.FloatTensor([reward2]))

            if time_step.last() == True:
                done = True
            else:
                done=False

            buffer_obs = np.stack([obs1.numpy().reshape(12),obs2.numpy().reshape(12)])
            buffer_next_obs = np.stack([next_obs1.numpy().reshape(12),next_obs2.numpy().reshape(12)])

            replay_buffer.push(buffer_obs, action, time_step.rewards, buffer_next_obs, done)

            i=i+1
            if done == True:
                print(t_eps, i, "done reached")
                if reward1>0:
                    print("a won")
                if reward2>0:
                    print("b won")
                break
    if reward1 > 0:
        writer.add_scalar('Steps/agent1', i, t_eps)
        writer.add_scalar('Steps/agent2', 0, t_eps)
    elif reward2 > 0:
        writer.add_scalar('Steps/agent1', 0, t_eps)
        writer.add_scalar('Steps/agent2', i, t_eps)
    else:
        writer.add_scalar('Steps/agent1', 0, t_eps)
        writer.add_scalar('Steps/agent2', 0, t_eps)

    sample = replay_buffer.sample(sample_size)
    target_policy1, policy1 = maddpg1.get_policy()
    target_policy2, policy2 = maddpg2.get_policy()

    if t_eps==0:
        torch.save(policy1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + "0.pth")
        torch.save(policy2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + "0.pth")

    maddpg1.update(sample, target_policy2,policy2,logger=writer,iter=t_eps)
    maddpg2.update(sample, target_policy1,policy1,logger=writer,iter=t_eps)
    maddpg1.update_all_targets()
    maddpg2.update_all_targets()

    if t_eps%2==0:
        critic1 = maddpg1.get_critic()
        critic2 = maddpg2.get_critic()
        torch.save(policy1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(policy2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")
        torch.save(critic1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/critic1_' + str(
                       t_eps) + ".pth")
        torch.save(critic2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/critic2_' + str(
                       t_eps) + ".pth")