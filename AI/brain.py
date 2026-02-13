import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# main brain
class policyNetwork(nn.Module):
    """nueral network for trading"""
    def __init__ (self, stateSize, actionSize):
        super().__init__()
    def __init__(self, stateSize, actionSize):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(stateSize, 128),   # Input layer  wide to capture feature combos
            nn.ReLU(),
            nn.Linear(128, 128),         # Hidden layer  learn complex patterns
            nn.ReLU(),
            nn.Linear(128, 64),          # Hidden layer  compress patterns
            nn.ReLU(),
            nn.Linear(64, 32),           # Extra hidden layer  fine-grain compression
            nn.ReLU(),
            nn.Linear(32, actionSize)    # Output layer  Q-values for actions
        )

    def forward(self, x):
        return self.layer(x)
    


import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    def push(self, *args): self.memory.append(args)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

def trainDQN(env, episodes=4000, gamma=0.95, epsilon = 0.01, lr=1e-3, stateSize=9, actionSize=3, resume = False):
    policy = policyNetwork(stateSize, actionSize).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    memory = ReplayBuffer(20000)
    batch_size = 64

    checkpoint_path = "AImodels/trading_checkpoint.pth"
    start_episode = 0 
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01

    # Loads half trained model if we want
    if resume and os.path.exists(checkpoint_path):
        print(f"--- Found Checkpoint! Resuming... ---")
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        print(f"--- Resuming from Episode {start_episode+1} with Epsilon {epsilon:.2f} ---")


    # loop through each episode
    for episode in range(start_episode, episodes):
        state = env.reset()
        totalReward = 0

        #Action tracker
        action_counts = {0: 0, 1: 0, 2: 0}
        
        #main simulatioan
        while not env.done:
            # Action Selection
            if np.random.rand() < epsilon:
                action = np.random.randint(actionSize)
            else:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = torch.argmax(policy(state_t)).item()
            
            action_counts[action] += 1

            next_state, reward, done = env.step(action)

            memory.push(state, action, reward, next_state, done)
            
            
            state = next_state
            totalReward += reward

            # 4. Training Step (The "Brain" Update)
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                
                # 1. Convert lists to single numpy arrays FIRST (This fixes the warning)
                # 2. Then convert those numpy arrays to torch tensors
                b_states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32).to(device)
                b_actions = torch.tensor([t[1] for t in transitions]).long().to(device)
                b_rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32).to(device)
                b_next_states = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32).to(device)
                b_dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32).to(device)

                # Current Q Values
                current_q = policy(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Target Q Values (Bellman Equation)
                with torch.no_grad():  
                    max_next_q = policy(b_next_states).max(1)[0]
                    target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))

                # Update Weights
                loss = lossFunc(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min: epsilon *= epsilon_decay

        if (episode + 1) % 15 == 0:
            torch.save({
                'episode': episode + 1,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }, checkpoint_path)
            print(f"--- Checkpoint saved at episode {episode+1} ---")


        final_price = env.df.iloc[env.t-1]["close"]
        final_value = env.balance + env.holdingNum * final_price
        profit = final_value - env.startBalance
        profit_pct = (profit / env.startBalance) * 100

        # ===== Console Output =====
        print(f"Episode {episode+1}")
        print(f"   Total Reward: {totalReward:.6f} | Epsilon: {epsilon:.2f}")
        print(f"   Final Portfolio: ${final_value:.2f}  |  Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        print(f"   Actions -> Hold: {action_counts[0]}, Buy: {action_counts[1]}, Sell: {action_counts[2]}")
            



    return policy
    
    
# def trainDQN(env, episodes = 1000, gamma=0.95, lr=1e-3, epsilon=0.1, stateSize = 5, actionSize = 3):
#     """trains the AI using reinforcement learning"""
#     policy = policyNetwork(stateSize, actionSize).to(device)
#     optimizer = optim.Adam(policy.parameters(), lr=lr)
#     lossFunc = nn.MSELoss()
    

#     epsilon = 1.0        # Start 100% random
#     epsilon_min = 0.01   # End 1% random
#     epsilon_decay = 0.995 # Multiply epsilon by this every episode

#     for episode in range(episodes):
#         state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(device)  # shape [1, stateSize]
#         totalReward = 0

#         if epsilon > epsilon_min:
#             epsilon *= epsilon_decay

#         while not env.done:
#             if np.random.rand() < epsilon:
#                 action = np.random.randint(actionSize)
#             else:
#                 qvals = policy(state)  # shape [1, actionSize]
#                 action = torch.argmax(qvals).item()

#             nextState, reward, done = env.step(action)
#             nextState_t = torch.tensor(nextState, dtype=torch.float32).unsqueeze(0).to(device)  # shape [1, stateSize]
#             totalReward += reward

#             # Compute target
#             with torch.no_grad():
#                 qNext = policy(nextState_t)           # [1, actionSize]
#                 maxQNext = torch.max(qNext, dim=1)[0] # [1]
#                 target = reward + gamma * maxQNext * (0 if done else 1)  # [1]

#             # Update Q-value
#             qPrediction = policy(state)[0, action].unsqueeze(0)  # shape [1]
#             loss = lossFunc(qPrediction, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             state = nextState_t

#         print(f"Episode {episode+1}/{episodes}, Total Reward: {totalReward}")

#     return policy
