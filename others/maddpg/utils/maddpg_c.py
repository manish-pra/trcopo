from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, policy, critic
# from .networks import policy_orca as policy #For orca one may use this network
# from .networks import critic_orca as critic #For orca one may use this network
from .misc import hard_update, gumbel_softmax, onehot_from_logits, soft_update
from .noise import OUNoise
import torch
MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic,
                 lr=0.01, discrete_action=True, agent_i=1):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = policy(num_in_pol,num_out_pol)
        self.critic = critic(num_in_critic,2*num_out_pol)
        self.target_policy = policy(num_in_pol,num_out_pol)
        self.target_critic = critic(num_in_critic,2*num_out_pol) # this take state and action dimension
        self.agent_i = agent_i -1

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        # self.critic.load_state_dict(torch.load("critic"+str(agent_i)+"_.pth"))
        # self.policy.load_state_dict(torch.load("agent"+str(agent_i)+"_.pth"))
        # self.target_critic.load_state_dict(torch.load("critic"+str(agent_i)+"_.pth"))
        # self.target_policy.load_state_dict(torch.load("agent"+str(agent_i)+"_.pth"))

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.tau = 0.01
        self.gamma=0.95

        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        # print('after policy',action)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
                # print('after gumbel',action)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += 0.3*torch.randn(action.shape)#Variable(Tensor(self.exploration.noise()),requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def get_policy(self):
        return self.target_policy, self.policy

    def get_critic(self):
        return self.critic

    def update(self, sample, oppo_target_policy, oppo_policy, parallel=False, logger=None,iter=5):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged

            In the code below, discrete is adapted for soccer and countinuous is for CarRacing
        """
        obs, acs, rews, next_obs, dones = sample

        self.critic_optimizer.zero_grad()
        # if self.alg_types[agent_i] == 'MADDPG':
        if self.discrete_action: # one-hot encode action
            if self.agent_i ==0:
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                    zip([self.target_policy,oppo_target_policy], next_obs)]
            else:
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                    zip([oppo_target_policy,self.target_policy], next_obs)]
            # all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
            #                     zip([self.target_policy,oppo_target_policy], next_obs)]
        else:
            if self.agent_i ==0:
                all_trgt_acs = [pi(nobs) for pi, nobs in
                                    zip([self.target_policy,oppo_target_policy], next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in
                                    zip([oppo_target_policy,self.target_policy], next_obs)]
            # all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policy,
            #                                              next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

        if self.discrete_action:
            target_value = (rews[self.agent_i].view(-1, 1) + self.gamma *
                            self.target_critic(trgt_vf_in) *
                            (1 - dones[self.agent_i].view(-1, 1))) #change after
        else:
            target_value = (rews[self.agent_i].view(-1, 1) + self.gamma *self.target_critic(trgt_vf_in)*(dones.view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = self.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()

        if self.discrete_action:
            curr_pol_out = self.policy(obs[self.agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.policy(obs[self.agent_i])
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        if self.discrete_action:
            if self.agent_i == 0:
                all_pol_acs.append(curr_pol_vf_in)
                all_pol_acs.append(onehot_from_logits(oppo_policy(obs[1])))
            else:
                all_pol_acs.append(onehot_from_logits(oppo_policy(obs[0])))
                all_pol_acs.append(curr_pol_vf_in)
        else:
            if self.agent_i == 0:
                all_pol_acs.append(curr_pol_vf_in)
                all_pol_acs.append(oppo_policy(obs[1]))
            else:
                all_pol_acs.append(oppo_policy(obs[0]))
                all_pol_acs.append(curr_pol_vf_in)

        #
        # for i, ob in zip(range(self.nagents), obs):
        #     if i == self.agent_i-1:
        #         all_pol_acs.append(curr_pol_vf_in)
        #     elif self.discrete_action:
        #         all_pol_acs.append(onehot_from_logits(self.policy(ob)))
        #     else:
        #         all_pol_acs.append(self.policy(ob))

        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -self.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        total_norm=0
        for p in self.policy.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_policy, self.policy, self.tau)

