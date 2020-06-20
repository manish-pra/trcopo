import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from trcopo_optim import TRPO
from torch.distributions import Categorical
import numpy as np
from MatchingPennies.matching_pennies import pennies_game
from torch.utils.tensorboard import SummaryWriter
from MatchingPennies.network import policy1, policy2
import os

folder_location = 'tensorboard/mp/'
experiment_name = 'trgda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)


writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
p1 = policy1()
p2 = policy2()

optim_p1 = TRPO(p1, lam = 1, bound=1e-4, esp=0.001)
optim_p2 = TRPO(p2, lam = 1, bound=1e-4, esp=0.001)

batch_size = 400
num_episode = 1000

env = pennies_game()

for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    state, _, _, _, _ = env.reset()
    #data_collection
    for i in range(1000):
        pi1 = p1()
        dist1 = Categorical(pi1)
        action1 = dist1.sample()

        pi2 = p2()
        dist2 = Categorical(pi2)
        action2 = dist2.sample()
        action = np.array([action1, action2])

        state = np.array([0,0])
        mat_state1.append(torch.FloatTensor(state))
        mat_state2.append(torch.FloatTensor(state))
        mat_action.append(torch.FloatTensor(action))
        #print(action)

        state, reward1, reward2, done, _ = env.step(action)

        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))
        mat_done.append(torch.FloatTensor([1 - done]))

    #print(action)
    # print('a1',dist1.mean, dist1.variance)
    # print('a2',dist2.mean, dist2.variance)
    action_both = torch.stack(mat_action)
    writer.add_scalar('Entropy/Agent1', dist1.entropy().data, t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy().data, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    #val1_p = -advantage_mat1#val1.detach()
    val1_p = torch.stack(mat_reward1).transpose(0,1)
    # st_time = time.time()
    # calculate gradients
    if val1_p.size(0)!=1:
        raise 'error'

    action_both = torch.stack(mat_action)

    def get_logprob1():
        pi_a1_s = p1()
        dist_pi1 = Categorical(pi_a1_s)
        log_probs1 = dist_pi1.log_prob(action_both[:, 0])
        return log_probs1, val1_p

    def get_logprob2():
        pi_a2_s = p2()
        dist_pi2 = Categorical(pi_a2_s)
        log_probs2 = dist_pi2.log_prob(action_both[:, 1])
        return log_probs2, -val1_p

    optim_p1.zero_grad()
    lam, improvement, a , sAs = optim_p1.step(get_logprob1)
    writer.add_scalar('Improvement/agent1', improvement, t_eps)
    writer.add_scalar('lambda/agent1', lam, t_eps)
    writer.add_scalar('ratio/agent1', a, t_eps)
    writer.add_scalar('ratio/sAs_a1', sAs, t_eps)

    norm_gx,norm_cgx, itr, norm_kl = optim_p1.getinfo()
    writer.add_scalar('debug/A1_norm_gx', norm_gx, t_eps)
    writer.add_scalar('debug/A1_norm_cx', norm_cgx, t_eps)
    writer.add_scalar('debug/A1_norm_kl', norm_kl, t_eps)
    writer.add_scalar('debug/A1_inv_itr', itr, t_eps)

    optim_p2.zero_grad()
    lam, improvement, a , sAs = optim_p2.step(get_logprob2)
    writer.add_scalar('Improvement/agent2', improvement, t_eps)
    writer.add_scalar('lambda/agent2', lam, t_eps)
    writer.add_scalar('ratio/agent2', a, t_eps)
    writer.add_scalar('ratio/sAs_a2', sAs, t_eps)


    norm_gx,norm_cgx, itr, norm_kl = optim_p2.getinfo()
    writer.add_scalar('debug/A2_norm_gx', norm_gx, t_eps)
    writer.add_scalar('debug/A2_norm_cx', norm_cgx, t_eps)
    writer.add_scalar('debug/A2_norm_kl', norm_kl, t_eps)
    writer.add_scalar('debug/A2_inv_itr', itr, t_eps)


    if t_eps%100 ==0:
        for p in p1.parameters():
            print('p1', p)
        for p in p2.parameters():
            print('p2', p)

    for p in p1.parameters():
        writer.add_scalar('Agent1/p1', p.data[0], t_eps)
        writer.add_scalar('Agent1/p2', p.data[1], t_eps)

    for p in p2.parameters():
        writer.add_scalar('Agent2/p1', p.data[0], t_eps)
        writer.add_scalar('Agent2/p2', p.data[1], t_eps)
            #print('p2', p)
    writer.add_scalar('Agent1/sm1', pi1.data[0], t_eps)
    writer.add_scalar('Agent1/sm2', pi1.data[1], t_eps)
    writer.add_scalar('Agent2/sm1', pi2.data[0], t_eps)
    writer.add_scalar('Agent2/sm2', pi2.data[1], t_eps)

    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")