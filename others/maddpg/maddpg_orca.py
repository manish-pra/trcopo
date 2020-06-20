# Game imports
import sys
import os
sys.path.insert(0, '../..')
import torch
from others.maddpg.buffer import ReplayBuffer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from others.maddpg.utils.maddpg_c import MADDPG
import random

import json
import sys

import car_racing_simulator.VehicleModel as VehicleModel
import car_racing_simulator.Track as Track
from CarRacing.orca_env_function import getreward, getdone, getfreezereward, getfreezecollosionreward, getfreezecollosionReachedreward, getfreezeTimecollosionReachedreward
import gc

folder_location = 'tensorboard/orca/'
experiment_name = 'maddpg/'

directory = '../' + folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
config = json.load(open('config.json'))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cpu'

vehicle_model = VehicleModel.VehicleModel(config["n_batch"], 'cpu', config)

x0 = torch.zeros(config["n_batch"], config["n_state"])

u0 = torch.zeros(config["n_batch"], config["n_control"])

maddpg1 = MADDPG(num_in_pol=10, num_out_pol=2, num_in_critic=20,lr = 0.000005,discrete_action=False,agent_i=1)
maddpg2 = MADDPG(num_in_pol=10, num_out_pol=2, num_in_critic=20,lr = 0.000005,discrete_action=False,agent_i=2)


batch_size = 5
num_episode = 10000
sample_size = 600
buffer_length=1800
num_ppo_epochs = 10
size_mini_batch=4
num_players=2

replay_buffer = ReplayBuffer(buffer_length, num_players,
                             [12,12],
                             [5,5])

for t_eps in range(num_episode):
    mat_action1 = []
    mat_action2 = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    print(t_eps)

    #data_collection
    avg_itr = 0

    curr_batch_size = 5

    #state = torch.zeros(config["n_batch"], config["n_state"])
    state_c1 = torch.zeros(curr_batch_size, config["n_state"])#state[:,0:6].view(6)
    state_c2 = torch.zeros(curr_batch_size, config["n_state"])#state[:, 6:12].view(6)
    init_p1 = torch.zeros((curr_batch_size)) #5*torch.rand((curr_batch_size))
    init_p2 = torch.zeros((curr_batch_size)) #5*torch.rand((curr_batch_size))
    state_c1[:,0] = init_p1
    state_c2[:,0] = init_p2
    a = random.choice([-0.1,0.1])
    b = a*(-1)
    state_c1[:, 1] = a*torch.ones((curr_batch_size))
    state_c2[:, 1] = b*torch.ones((curr_batch_size))
    batch_mat_state1 = torch.empty(0)
    batch_mat_state2 = torch.empty(0)
    batch_mat_action1 = torch.empty(0)
    batch_mat_action2 = torch.empty(0)
    batch_mat_reward1 = torch.empty(0)
    batch_mat_done = torch.empty(0)

    itr = 0
    done = torch.tensor([False])
    done_c1 = torch.zeros((curr_batch_size)) <= -0.1
    done_c2 = torch.zeros((curr_batch_size)) <= -0.1
    prev_coll_c1 = torch.zeros((curr_batch_size)) <= -0.1
    prev_coll_c2 = torch.zeros((curr_batch_size)) <= -0.1
    counter1 = torch.zeros((curr_batch_size))
    counter2 = torch.zeros((curr_batch_size))

    #for itr in range(50):
    while np.all(done.numpy()) == False:
        avg_itr+=1

        obs1 = torch.cat([state_c1[:, 0:5], state_c2[:, 0:5]], dim=1)
        action1 = maddpg1.step(obs1,explore=True)
        obs2 = torch.cat([state_c2[:, 0:5], state_c1[:, 0:5]], dim=1)
        action2 = maddpg2.step(obs2,explore=True)

        # dist1 = p1(st1_gpu)
        # action1 = dist1.sample().to('cpu')
        #
        # st2_gpu = torch.cat([state_c2[:, 0:5], state_c1[:, 0:5]], dim=1).to(device)
        #
        # dist2 = p2(st2_gpu)
        # action2 = dist2.sample().to('cpu')
        # action1 = torch.Tensor([[1,1],[1,1]])
        # action2 = torch.Tensor([[1,1],[1,1]])

        if itr>0:
            mat_state1 = torch.cat([mat_state1.view(-1,curr_batch_size,5),state_c1[:,0:5].view(-1,curr_batch_size,5)],dim=0) # concate along dim = 0
            mat_state2 = torch.cat([mat_state2.view(-1, curr_batch_size, 5), state_c2[:, 0:5].view(-1, curr_batch_size, 5)], dim=0)
            mat_action1 = torch.cat([mat_action1.view(-1, curr_batch_size, 2), action1.view(-1, curr_batch_size, 2)], dim=0)
            mat_action2 = torch.cat([mat_action2.view(-1, curr_batch_size, 2), action2.view(-1, curr_batch_size, 2)], dim=0)
        else:
            mat_state1 = state_c1[:,0:5]
            mat_state2 = state_c2[:, 0:5]
            mat_action1 = action1
            mat_action2 = action2
        #mat_state2.append(state_c2[:,0:5])
        # mat_action1.append(action1)
        # mat_action2.append(action2)

        prev_state_c1 = state_c1
        prev_state_c2 = state_c2

        state_c1 = vehicle_model.dynModelBlendBatch(state_c1.view(-1,6), action1.view(-1,2)).view(-1,6)
        state_c2 = vehicle_model.dynModelBlendBatch(state_c2.view(-1,6), action2.view(-1,2)).view(-1,6)

        state_c1 = (state_c1.transpose(0, 1) * (~done_c1) + prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)
        state_c2 = (state_c2.transpose(0, 1) * (~done_c2) + prev_state_c2.transpose(0, 1) * (done_c2)).transpose(0, 1)

        reward1, reward2, done_c1, done_c2, coll_c1, coll_c2, counter1, counter2 = getfreezeTimecollosionReachedreward(state_c1, state_c2,
                                                                     vehicle_model.getLocalBounds(state_c1[:, 0]),
                                                                     vehicle_model.getLocalBounds(state_c2[:, 0]),
                                                                     prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2, counter1, counter2)

        done = (done_c1) * (done_c2)  # ~((~done_c1) * (~done_c2))
        # done =  ~((~done_c1) * (~done_c2))
        mask_ele = ~done

        if itr>0:
            mat_reward1 = torch.cat([mat_reward1.view(-1,curr_batch_size,1),reward1.view(-1,curr_batch_size,1)],dim=0) # concate along dim = 0
            mat_done = torch.cat([mat_done.view(-1, curr_batch_size, 1), mask_ele.view(-1, curr_batch_size, 1)], dim=0)
        else:
            mat_reward1 = reward1
            mat_done = mask_ele

        remaining_xo = ~done

        state_c1 = state_c1[remaining_xo]
        state_c2 = state_c2[remaining_xo]
        prev_coll_c1 = coll_c1[remaining_xo]#removing elements that died
        prev_coll_c2 = coll_c2[remaining_xo]#removing elements that died
        counter1 = counter1[remaining_xo]
        counter2 = counter2[remaining_xo]

        curr_batch_size = state_c1.size(0)

        if curr_batch_size<remaining_xo.size(0):
            if batch_mat_action1.nelement() == 0:
                batch_mat_state1 = mat_state1.transpose(0, 1)[~remaining_xo].view(-1, 5)
                batch_mat_state2 = mat_state2.transpose(0, 1)[~remaining_xo].view(-1, 5)
                batch_mat_action1 = mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_action2 = mat_action2.transpose(0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_reward1 = mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)
                batch_mat_done = mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)
                # progress_done1 = batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[0, 0]
                # progress_done2 = batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[0, 0]
                progress_done1 = torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:,mat_state1.size(0)-1,0] - mat_state1.transpose(0, 1)[~remaining_xo][:,0,0])
                progress_done2 = torch.sum(mat_state2.transpose(0, 1)[~remaining_xo][:,mat_state2.size(0)-1,0] - mat_state2.transpose(0, 1)[~remaining_xo][:,0,0])
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]
            else:
                prev_size = batch_mat_state1.size(0)
                batch_mat_state1 = torch.cat([batch_mat_state1,mat_state1.transpose(0, 1)[~remaining_xo].view(-1,5)],dim=0)
                batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1)[~remaining_xo].view(-1, 5)],dim=0)
                batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)],dim=0)
                batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1)[~remaining_xo].view(-1, 2)],dim=0)
                batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[~remaining_xo][:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[~remaining_xo][:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[~remaining_xo][:, 0, 0])
                # progress_done1 = progress_done1 + batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[prev_size, 0]
                # progress_done2 = progress_done2 + batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[prev_size, 0]
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]

            mat_state1 = mat_state1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_state2 = mat_state2.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_action1 = mat_action1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_action2 = mat_action2.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_reward1 = mat_reward1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_done = mat_done.transpose(0, 1)[remaining_xo].transpose(0, 1)

        # print(avg_itr,remaining_xo.size(0))

        # writer.add_scalar('Reward/agent1', reward1, t_eps)
        itr = itr + 1

        if np.all(done.numpy()) == True or batch_mat_state1.size(0)>1000 or itr>250:# or itr>900: #brak only if all elements in the array are true
            prev_size = batch_mat_state1.size(0)
            batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1).reshape(-1, 1)],dim=0) #should i create a false or true array?
            print("done", itr)
            print(mat_done.shape)
            mat_done[mat_done.size(0)-1,:,:] = torch.ones((mat_done[mat_done.size(0)-1,:,:].shape))>=2 # creating a true array of that shape
            print(mat_done.shape, batch_mat_done.shape)
            if batch_mat_done.nelement() == 0:
                batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
                progress_done1 = 0
                progress_done2 =0
            else:
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
            if prev_size == batch_mat_state1.size(0):
                progress_done1 = progress_done1
                progress_done2 = progress_done2
            else:
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[:, 0, 0])
            print(batch_mat_done.shape)
            # print("done", itr)
            break

    # print(avg_itr)
    cast = lambda x: torch.autograd.Variable(torch.Tensor(x), requires_grad=False)

    # idx = np.random.randint(0, batch_mat_state1.size(0)-1, sample_size)
    if batch_mat_state1.size(0)-1>600:
        N_size = 600
    else:
        N_size = batch_mat_state1.size(0)-1

    idx = np.random.choice(np.arange(batch_mat_state1.size(0)-1), size=N_size, replace=False)

    # buffer_obs = [cast(batch_mat_state1[0:batch_mat_state1.size(0)-1][idx]),cast(batch_mat_state2[0:batch_mat_state1.size(0)-1][idx])]
    # buffer_next_obs = [cast(batch_mat_state1[1:batch_mat_state1.size(0)][idx]),cast(batch_mat_state2[1:batch_mat_state1.size(0)][idx])]

    st1 = torch.cat([batch_mat_state1[0:batch_mat_state1.size(0)-1][idx],batch_mat_state2[0:batch_mat_state1.size(0)-1][idx]],dim=1)
    st2 = torch.cat([batch_mat_state2[0:batch_mat_state1.size(0)-1][idx],batch_mat_state1[0:batch_mat_state1.size(0)-1][idx]],dim=1)
    next_st1 = torch.cat([batch_mat_state1[1:batch_mat_state1.size(0)][idx],batch_mat_state2[1:batch_mat_state1.size(0)][idx]],dim=1)
    next_st2 = torch.cat([batch_mat_state2[1:batch_mat_state1.size(0)][idx],batch_mat_state1[1:batch_mat_state1.size(0)][idx]],dim=1)

    buffer_obs = [cast(st1),cast(st2)]
    buffer_next_obs = [cast(next_st1),cast(next_st2)]

    buffer_action = [cast(batch_mat_action1[0:batch_mat_action1.size(0)-1][idx]),cast(batch_mat_action2[0:batch_mat_action1.size(0)-1][idx])]
    buffer_reward = [cast(batch_mat_reward1[0:batch_mat_reward1.size(0)-1][idx]), cast(-batch_mat_reward1[0:batch_mat_reward1.size(0)-1][idx])]
    #np.stack([next_obs1.numpy().reshape(12), next_obs2.numpy().reshape(12)])
    buffer_done = batch_mat_done[1:batch_mat_done.size(0)][idx]

    sample = buffer_obs, buffer_action, buffer_reward, buffer_next_obs, buffer_done

    target_policy1, policy1 = maddpg1.get_policy()
    target_policy2, policy2 = maddpg2.get_policy()
    maddpg1.update(sample, target_policy2, policy2, logger=writer, iter=t_eps)
    maddpg2.update(sample, target_policy1, policy1, logger=writer, iter=t_eps)
    maddpg1.update_all_targets()
    maddpg2.update_all_targets()

    print(batch_mat_state1.shape,itr)
    # writer.add_scalar('Dist/variance_throttle_p1', dist1.variance[0,0], t_eps)
    # writer.add_scalar('Dist/variance_steer_p1', dist1.variance[0,1], t_eps)
    # writer.add_scalar('Dist/variance_throttle_p2', dist2.variance[0,0], t_eps)
    # writer.add_scalar('Dist/variance_steer_p2', dist2.variance[0,1], t_eps)
    writer.add_scalar('Reward/mean', batch_mat_reward1.mean(), t_eps)
    writer.add_scalar('Reward/sum', batch_mat_reward1.sum(), t_eps)
    writer.add_scalar('Progress/final_p1', progress_done1/batch_size, t_eps)
    writer.add_scalar('Progress/final_p2', progress_done2/batch_size, t_eps)
    writer.add_scalar('Progress/trajectory_length', itr, t_eps)
    writer.add_scalar('Progress/agent1', batch_mat_state1[:,0].mean(), t_eps)
    writer.add_scalar('Progress/agent2', batch_mat_state2[:,0].mean(), t_eps)


    if t_eps%20==0:
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