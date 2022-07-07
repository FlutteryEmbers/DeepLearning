import numpy as np
import torch as T
from networks.ppo_net import Actor, Critic

class ReplayBuffer():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.state_values = []
        self.log_pi = []
    
    def store(self, state, action, next_state, reward, done, state_value, log_pi):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)
        self.state_values.append(state_value)
        self.log_pi.append(log_pi)

    def sample(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.next_states), \
            np.array(self.rewards), np.array(self.dones), np.array(self.state_values), \
                np.array(self.log_pi), batches  

    def clear(self):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.state_values = []
        self.log_pi = []

class Agent():
    def __init__(self, env, n_inputs, n_actions, actor_lr=1e-5, critics_lr=1e-4, gamma=0.99, GAE_lambda=0.95, clip=0.2, batch_size=32, n_epochs=10) -> None:
        self.n_epochs= n_epochs
        self.gamma = gamma
        self.GAE_lambda = GAE_lambda
        self.policy_clip = clip
        self.env = env

        self.actor = Actor(n_inputs=n_inputs, n_actions=n_actions, lr=actor_lr)
        self.critic = Critic(n_inputs=n_inputs, lr=critics_lr)

        self.buffer_size = batch_size
        self.replay_buffer = ReplayBuffer(batch_size)

    def store(self, state, action, next_state, reward, done, state_value, log_pi):
        self.replay_buffer.store(state=state, action=action, next_state=next_state, reward=reward, done=done, state_value=state_value, log_pi=log_pi)

    def choose_action(self, state):
        state = T.tensor(np.array(state))
        action_distribution = self.actor(state)
        action = action_distribution.sample()
        log_val = action_distribution.log_prob(action)
        state_value = self.critic(state)

        return action.item(), log_val.item(), state_value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, next_states, rewards, dones, state_values, log_pi, batches = self.replay_buffer.sample()

            advantages = np.zeros(len(rewards), dtype=np.float32)
            for t in range(len(rewards)-1):
                discount_factor = 1
                a_t = 0
                for j in range(t, len(rewards) - 1):
                    delta_t = rewards[t] + self.gamma * state_values[j+1]*(1-dones[j]) - state_values[j]
                    a_t = a_t + discount_factor * delta_t
                    discount_factor = discount_factor * self.gamma * self.GAE_lambda
                advantages[t] = a_t

            advantages = T.tensor(advantages).to(self.actor.device)
            state_values = T.tensor(state_values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(states[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(log_pi, dtype=T.float).to(self.actor.device)
                actions = T.tensor(actions, dtype=T.float).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                
                critic_value = T.squeeze(critic_value)
                #  dist = T.squeeze(dist)

                new_prob = dist.log_prob(actions)
                prob_ratio = new_prob.exp()/old_probs.exp()

                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)*advantages[batch]
                actor_loss = - T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantages[batch] + state_values[batch]
                critic_loss = (returns - critic_value) **2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.replay_buffer.clear()

    def run(self, n_games, N = 20):
        scores = []
        n_steps = 0
        for i in range(n_games):
            score = 0
            state = self.env.reset()
            done = False
            # agent.reset()
            while not done:
                # action = agent.choose_action(state)
                action, prob, val = self.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                n_steps += 1
                score += reward
                self.env.render()
                self.store(state=state, action=action, next_state=state_, reward=reward, done=done, state_value=val, log_pi=prob)
                if n_steps % N == 0:
                    self.learn()
                
                state = state_

            # agent.learn()
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('episode', i, 'score %.1f' % score, 'average score %.2f' % avg_score)

        self.env.close()







