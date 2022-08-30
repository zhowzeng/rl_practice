import torch
from torch import nn


class VanillaPolicyGradient:
    def __init__(self, env, gamma: float = 0.99):
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0]
        self.policy = nn.Sequential(
            [nn.Linear(self.n_features, 64), nn.Tanh(), nn.Linear(64, self.n_actions)]
        )

    def learn(self):
        ...
        # probs = policy_network(state)
        # m = Categorical(probs)
        # action = m.sample()
        # log_prob = m.log_prob(action)

        # for i_episode in range(3000):
        #     observation = env.reset()
        #     while True:
        #         if RENDER: env.render()
        #         action = RL.choose_action(observation)
        #         observation_, reward, done, info = env.step(action)
        #         RL.store_transition(observation, action, reward)
        #         if done:
        #             ep_rs_sum = sum(RL.ep_rs)
        #             if 'running_reward' not in globals():
        #                 running_reward = ep_rs_sum
        #             else:
        #                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
        #             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
        #             print("episode:", i_episode, "  reward:", int(running_reward))
        #             vt = RL.learn()
        #             if i_episode == 0:
        #                 plt.plot(vt)    # plot the episode vt
        #                 plt.xlabel('episode steps')
        #                 plt.ylabel('normalized state-action value')
        #                 plt.show()
        #             break
        #         observation = observation_
