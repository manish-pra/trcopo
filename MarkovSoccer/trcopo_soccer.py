# Game imports
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from open_spiel.python import rl_environment

import torch
from trcopo_optim import TRCoPO
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from MarkovSoccer.networks import policy
from MarkovSoccer.networks import critic

from trcopo_optim.critic_functions import critic_update, get_advantage
from MarkovSoccer.soccer_state import get_relative_state, get_two_state
import time

folder_location = 'tensorboard/soccer/'
experiment_name = 'trcopo/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)

writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
p1 = policy(12,5)
p2 = policy(12,5)
q = critic(12)

optim_q = torch.optim.Adam(q.parameters(), lr=0.001)

optim = TRCoPO(p1.parameters(),p2.parameters(), threshold=0.0001)

batch_size = 10
num_episode = 10001

game = "markov_soccer"
num_players = 2

env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]

for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    time_step = env.reset()
    # print(pretty_board(time_step))
    a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

    #data_collection
    avg_itr = 0
    for _ in range(batch_size):
        i = 0
        while time_step.last()==False:
            action1_prob = p1(torch.FloatTensor(rel_state1))
            #print('a1',action1_prob)
            dist1 = Categorical(action1_prob)
            action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions
            avg_itr = avg_itr + 1

            action2_prob = p2(torch.FloatTensor(rel_state2))
            #print('a2',action2_prob)
            dist2 = Categorical(action2_prob)
            action2 = dist2.sample()

            action = np.array([action1, action2])

            mat_state1.append(torch.FloatTensor(rel_state1))
            mat_state2.append(torch.FloatTensor(rel_state2))
            mat_action.append(torch.FloatTensor(action))

            time_step = env.step(action)
            a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

            reward1 = time_step.rewards[0]
            reward2 = time_step.rewards[1]

            #print(pretty_board(time_step))
            mat_reward1.append(torch.FloatTensor([reward1]))
            mat_reward2.append(torch.FloatTensor([reward2]))

            if time_step.last() == True:
                done = True
            else:
                done=False

            mat_done.append(torch.FloatTensor([1 - done]))
            i=i+1
            if done == True:
                print(t_eps, i, "done reached")
                if reward1>0:
                    print("a won")
                if reward2>0:
                    print("b won")
                time_step = env.reset()
                # print(pretty_board(time_step))
                a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
                #print(state1, state2, reward1,reward2, done)
                break

    if reward1>0:
        writer.add_scalar('Steps/agent1', i, t_eps)
        writer.add_scalar('Steps/agent2', 50, t_eps)
    elif reward2>0:
        writer.add_scalar('Steps/agent1', 50, t_eps)
        writer.add_scalar('Steps/agent2', i, t_eps)
    else:
        writer.add_scalar('Steps/agent1', 50, t_eps)
        writer.add_scalar('Steps/agent2', 50, t_eps)

    val1 = q(torch.stack(mat_state1))
    val1 = val1.detach()
    next_value = 0
    returns_np1 = get_advantage(next_value, torch.stack(mat_reward1), val1, torch.stack(mat_done), gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1 - val1.transpose(0,1)

    val2 = q(torch.stack(mat_state2))
    val2 = val2.detach()
    next_value = 0
    returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)

    returns2 = torch.cat(returns_np2)
    advantage_mat2 = returns2 - val2.transpose(0,1)

    for loss_critic, gradient_norm in critic_update(torch.cat([torch.stack(mat_state1),torch.stack(mat_state2)]), torch.cat([returns1,returns2]).view(-1,1), q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)

    ed_q_time = time.time()

    optim.zero_grad()

    def get_log_prob():
        pi_a1_s = p1(torch.stack(mat_state1))
        dist_batch1 = Categorical(pi_a1_s)
        action_both = torch.stack(mat_action)
        log_probs1 = dist_batch1.log_prob(action_both[:, 0])
        pi_a2_s = p2(torch.stack(mat_state2))
        dist_batch2 = Categorical(pi_a2_s)
        log_probs2 = dist_batch2.log_prob(action_both[:, 1])
        return log_probs1, log_probs2, advantage_mat1

    improve1, improve2, lamda, lam1, lam2, esp, stat, gmres_itr = optim.step(get_log_prob)

    writer.add_scalar('Improvement/agent1', improve1, t_eps)
    writer.add_scalar('Improvement/agent2', improve2, t_eps)
    writer.add_scalar('Improvement/error', esp, t_eps)
    writer.add_scalar('Improvement/status', stat, t_eps)

    writer.add_scalar('lamda/agent1', lam1, t_eps)
    writer.add_scalar('lamda/agent2', lam2, t_eps)
    writer.add_scalar('lamda/commona', lamda, t_eps)


    _, _, cgx, cgy, itr_num = optim.getinfo()
    writer.add_scalar('norm/theta_diff', cgx + cgy, t_eps)
    writer.add_scalar('norm/itr_num', itr_num, t_eps)
    writer.add_scalar('norm/gmres_itr', gmres_itr, t_eps)

    writer.add_scalar('Entropy/agent1', dist1.entropy().mean().detach(), t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy().mean().detach(), t_eps)

    if t_eps%10==0:
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")
        torch.save(q.state_dict(),
                   '../' + folder_location + experiment_name + 'model/val_' + str(
                       t_eps) + ".pth")