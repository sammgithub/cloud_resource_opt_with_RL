########################################
######## SERAJ A MOSTAFA ###############
### PhD Candiate, IS Dept. UMBC ########
########################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time, os, copy
import pandas as pd
from datetime import datetime

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Environment and Replay Buffer ====================

class CloudEnv:
    """
    A simulated cloud environment.
    State: vector of [cpu_util, mem_util, gpu_util, queue_length, sla_violation]
    Action: discrete choices (e.g., 0: scale down, 1: no change, 2: scale up)
    Reward: vector reward computed from cost, latency, and resource efficiency.
    """
    def __init__(self):
        self.state_dim = 5
        self.action_space = [0,1,2]  # scale down, unchanged, scale up
        self.current_state = self.reset()
    
    def reset(self):
        # Initialize state randomly; in practice, use your simulation parameters.
        self.current_state = np.random.rand(self.state_dim)
        return self.current_state
    
    def step(self, action):
        # Simple simulation: update state based on action
        # For example, action=2 (scale up) may decrease queue length but increase cost.
        s = self.current_state.copy()
        if action == 0:
            s[0] = max(0, s[0] - 0.1)  # lower CPU utilization
            s[3] = min(1, s[3] + 0.1)   # higher queue length
        elif action == 2:
            s[0] = min(1, s[0] + 0.1)   # higher CPU utilization
            s[3] = max(0, s[3] - 0.1)    # lower queue length
        # Simulate cost and SLA penalty as part of reward computation
        cost = 1.0 - s[0]  # lower CPU utilization means lower cost
        sla_penalty = s[4] * 10.0  # high SLA_violation => high penalty
        # Combined reward (for scalarized training)
        reward = - cost - sla_penalty
        # Randomly update other parts of state
        s[1] = np.random.rand()
        s[2] = np.random.rand()
        s[4] = np.random.rand()  # random SLA violation indicator
        self.current_state = s
        done = False
        return s, reward, done, {}

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.storage = []
    
    def add(self, transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
        self.storage.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.storage), batch_size, replace=False)
        batch = [self.storage[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32, device=device),
                torch.tensor(actions, dtype=torch.long, device=device),
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.tensor(next_states, dtype=torch.float32, device=device))
    
    def size(self):
        return len(self.storage)

# ==================== Neural Network Models ====================

def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        # For discrete actions, assume action is one-hot encoded.
        x = torch.cat([state, action], dim=1)
        return self.net(x)

def copy_network(net):
    return copy.deepcopy(net)

# ==================== Adaptive Critic Reset ====================

def reset_critics(critic1, critic2):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    critic1.apply(init_weights)
    critic2.apply(init_weights)
    print("Critic networks have been reset.")

# ==================== Main Training Loop ====================

def train_rl(num_episodes=1000, batch_size=64, pretrain_steps=1000, reset_interval=200, p_reset=0.1):
    # Hyperparameters
    gamma = 0.99
    tau = 0.005
    learning_rate = 1e-3
    state_dim = 5
    action_dim = 3

    # Initialize environment, replay buffers
    env = CloudEnv()
    offline_buffer = ReplayBuffer(max_size=10000)
    online_buffer = ReplayBuffer(max_size=10000)
    
    # For demonstration, load offline data from CSV files if available.
    # Here we simulate by filling the offline buffer with random transitions.
    for _ in range(5000):
        s = env.reset()
        a = random.choice(env.action_space)
        s_next, r, done, _ = env.step(a)
        offline_buffer.add((s, a, r, s_next))
    
    # Initialize networks
    actor = Actor(state_dim, action_dim).to(device)
    critic1 = Critic(state_dim, action_dim).to(device)
    critic2 = Critic(state_dim, action_dim).to(device)
    target_critic1 = copy_network(critic1).to(device)
    target_critic2 = copy_network(critic2).to(device)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(list(critic1.parameters())+list(critic2.parameters()), lr=learning_rate)
    
    total_steps = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(200):  # max steps per episode
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action_probs = actor(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            next_state, reward, done, _ = env.step(action)
            online_buffer.add((state, action, reward, next_state))
            episode_reward += reward
            state = next_state
            total_steps += 1
            
            # Sample a mixed batch from offline and online buffers
            if online_buffer.size() > batch_size and offline_buffer.size() > batch_size:
                mix_ratio = 0.5
                num_offline = int(batch_size * mix_ratio)
                num_online = batch_size - num_offline
                b_states_off, b_actions_off, b_rewards_off, b_next_off = offline_buffer.sample(num_offline)
                b_states_on, b_actions_on, b_rewards_on, b_next_on = online_buffer.sample(num_online)
                b_states = torch.cat([b_states_off, b_states_on], dim=0)
                b_actions = torch.cat([b_actions_off, b_actions_on], dim=0)
                b_rewards = torch.cat([b_rewards_off, b_rewards_on], dim=0)
                b_next_states = torch.cat([b_next_off, b_next_on], dim=0)
                
                # One-hot encode actions
                b_actions_oh = one_hot(b_actions, action_dim)
                with torch.no_grad():
                    next_action_probs = actor(b_next_states)
                    next_actions = torch.multinomial(next_action_probs, num_samples=1).squeeze(-1)
                    next_actions_oh = one_hot(next_actions, action_dim)
                    Q1_next = target_critic1(b_next_states, next_actions_oh).squeeze(-1)
                    Q2_next = target_critic2(b_next_states, next_actions_oh).squeeze(-1)
                    min_Q_next = torch.min(Q1_next, Q2_next)
                    y = b_rewards + gamma * min_Q_next

                # Update critics
                Q1 = critic1(b_states, b_actions_oh).squeeze(-1)
                Q2 = critic2(b_states, b_actions_oh).squeeze(-1)
                critic_loss = nn.MSELoss()(Q1, y) + nn.MSELoss()(Q2, y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Update actor
                actor_loss = -torch.mean(critic1(b_states, actor(b_states)))
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks
                for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            # Adaptive critic reset: check every reset_interval steps
            if total_steps % reset_interval == 0:
                # For illustration, we reset with probability p_reset.
                if random.random() < p_reset:
                    reset_critics(critic1, critic2)
                    target_critic1 = copy_network(critic1)
                    target_critic2 = copy_network(critic2)
                    
            if done:
                break
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        # Periodically, save model checkpoints, log metrics, etc.
    
    # After training, evaluate the policy on a set of test episodes and generate plots.
    # (Evaluation code would be here, including functions to save results to CSV and produce LaTeX tables.)
    
    # Example: call external file (e.g., your vgg16 training) to compare logs.
    # You can integrate a function call such as:
    # os.system("python train-imagenet45-vgg16.py")
    
if __name__ == "__main__":
    train_rl()
