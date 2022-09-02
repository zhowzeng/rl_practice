import numpy as np
import torch
from gym import Env
from torch import nn
from torch.distributions import Categorical


class VanillaPolicyGradient:
    def __init__(self, env: Env, gamma=0.99, lr=1e-2):
        self.gamma = gamma
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0]
        self.policy = nn.Sequential(nn.Linear(self.n_features, 10), nn.Tanh(), nn.Linear(10, self.n_actions))
        self.env = env
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _discount_and_norm_rewards(self, eps_rew):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(eps_rew)
        running_add = 0
        for t in reversed(range(0, len(eps_rew))):
            running_add = running_add * self.gamma + eps_rew[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs

    def learn(self, episodes=100, verbose_step=5):
        for k in range(episodes):
            eps_rew = []
            eps_obs = []
            eps_act = []

            done = False
            obs = self.env.reset()
            while not done:
                logits = self.policy(torch.as_tensor(obs))
                distr = Categorical(logits=logits)
                act = distr.sample().item()
                obs, rew, done, _ = self.env.step(act)
                eps_rew.append(rew)
                eps_obs.append(obs)
                eps_act.append(act)

            self.optimizer.zero_grad()
            weights = torch.as_tensor(self._discount_and_norm_rewards(eps_rew))
            logits = self.policy(torch.as_tensor(np.array(eps_obs)))
            log_probs = Categorical(logits=logits).log_prob(torch.as_tensor(eps_act))
            eps_loss = -1 * (log_probs * weights).mean()

            eps_loss.backward()
            self.optimizer.step()

            if (k + 1) % verbose_step == 0:
                print(f'episode {k+1} loss {eps_loss.item()} return {sum(eps_rew)} length {len(eps_rew)}')
