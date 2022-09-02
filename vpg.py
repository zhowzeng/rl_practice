import numpy as np
import torch
from torch import nn


class VanillaPolicyGradient:
    def __init__(self, env, gamma=0.95, lr=1e-2):
        self.gamma = gamma
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0]
        self.env = env

        self.policy = nn.Sequential(nn.Linear(self.n_features, 10), nn.Tanh(), nn.Linear(10, self.n_actions))
        self.policy.apply(init_weights)

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

    def predict(self, obs):
        with torch.no_grad():
            logit = self.policy(torch.as_tensor(obs))
            return logit.argmax().item()

    def learn(self, episodes=100, verbose_step=5):
        for k in range(episodes):
            eps_rew = []
            eps_obs = []
            eps_act = []

            done = False
            obs = self.env.reset()
            while not done:
                eps_obs.append(obs)
                logits = self.policy(torch.as_tensor(obs))
                distr = torch.distributions.Categorical(logits=logits)
                act = distr.sample().item()
                obs, rew, done, _ = self.env.step(act)
                eps_rew.append(rew)
                eps_act.append(act)

            self.optimizer.zero_grad()
            weights = torch.as_tensor(self._discount_and_norm_rewards(eps_rew))
            logits = self.policy(torch.as_tensor(np.array(eps_obs)))
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(torch.as_tensor(eps_act))
            eps_loss = -1 * (log_probs * weights).mean()

            eps_loss.backward()
            self.optimizer.step()

            if (k + 1) % verbose_step == 0:
                print(f'episode {k+1} loss {eps_loss.item()} return {sum(eps_rew)}')

    def save(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load(self, model_path):
        state_dict = torch.load(model_path)
        self.policy.load_state_dict(state_dict)


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(0.0, 0.1)
        m.bias.data.fill_(0.1)
