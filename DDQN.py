# train_ddqn.py
# COMPLETE DDQN IN SINGLE FILE

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from preprocessing import build_snapshots
from environment import SpectrumEnv


# ===============================
# HYPERPARAMETERS
# ===============================

FILE_PATH = r"distance_snapshots.xlsx"

GAMMA = 0.99
LR = 5e-5 #
#LR = 3e-5 #More stable but slower convergence, may require more episodes to see improvement
BATCH_SIZE = 32 #we can increase this to 64 or 128 if we have enough memory, but it may slow down training
EPISODES = 500 

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995 #epsilon becomes 0.01 after ~300 episodes
#EPS_DECAY = 0.97 #epsilon becomes 0.01 after ~150 episodes

TARGET_UPDATE_FREQ = 50
BUFFER_SIZE = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Q NETWORK
# ===============================

class QNetwork(nn.Module):

    def __init__(self, state_dim=6, action_dim=4):
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,action_dim)
        )

    def forward(self,x):
        return self.model(x)


# ===============================
# REPLAY BUFFER
# ===============================

class ReplayBuffer:

    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):

        batch = random.sample(self.buffer,batch_size)

        states,actions,rewards,next_states,dones = zip(*batch)

        return(
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# ===============================
# BUILD ENVIRONMENT
# ===============================

snapshots = build_snapshots(FILE_PATH)

env = SpectrumEnv(snapshots)

state_dim = 6
action_dim = 4


q_net = QNetwork(state_dim,action_dim).to(device)
target_net = QNetwork(state_dim,action_dim).to(device)

target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(),lr=LR)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPS_START


# ===============================
# TRACK METRICS
# ===============================

episode_loss_history = []
throughput_history = []


# ===============================
# TRAINING LOOP
# ===============================

for episode in range(EPISODES):

    state = env.reset()
    done = False

    episode_losses = []

    while not done:

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # epsilon greedy

        if random.random() < epsilon:
            action = np.random.randint(action_dim)

        else:
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_state,reward,done = env.step(action)

        if next_state is None:
            next_state = np.zeros(state_dim)

        replay_buffer.push(state,action,reward,next_state,done)

        state = next_state


        # ===============================
        # TRAINING STEP (STOP AFTER CONVERGENCE)
        # ===============================

        if len(replay_buffer) > BATCH_SIZE:

            states,actions,rewards,next_states,dones = replay_buffer.sample(BATCH_SIZE)

            current_q = q_net(states).gather(1,actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():

                next_actions = torch.argmax(q_net(next_states),dim=1)

                next_q = target_net(next_states)\
                    .gather(1,next_actions.unsqueeze(1))\
                    .squeeze(1)

                target_q = rewards + GAMMA*next_q*(1-dones)

            loss = nn.SmoothL1Loss()(current_q,target_q)

            episode_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # ===============================
    # STORE LOSS
    # ===============================

    if len(episode_losses)>0:
        episode_loss_history.append(np.mean(episode_losses))
    else:
        episode_loss_history.append(episode_loss_history[-1] if episode_loss_history else 0)


    # ===============================
    # EPSILON DECAY
    # ===============================

    epsilon = max(EPS_END,epsilon*EPS_DECAY)


    # ===============================
    # TARGET NETWORK UPDATE
    # ===============================

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(q_net.state_dict())


    throughput_history.append(env.total_throughput)

    print(f"Episode {episode+1} | Throughput: {env.total_throughput:.2f} | Epsilon: {epsilon:.3f}")


print("\nTraining Completed.")


# ===============================
# SMOOTHING FUNCTION
# ===============================

def smooth_curve(data,window=40):

    smoothed=[]

    for i in range(len(data)):

        start=max(0,i-window)

        smoothed.append(np.mean(data[start:i+1]))

    return smoothed


# ===============================
# LOSS CURVE
# ===============================

smoothed_loss = smooth_curve(episode_loss_history,window=40)

plt.figure(figsize=(10,5))
plt.plot(smoothed_loss)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("DDQN Training Loss Curve")
plt.grid(True)
plt.show()


# ===============================
# THROUGHPUT CURVE
# ===============================

def average_curve(data,window=5):

    avg=[]
    x=[]

    for i in range(0,len(data),window):

        segment=data[i:i+window]

        avg.append(np.mean(segment))

        x.append(i)

    return x,avg


thr_x,thr_avg = average_curve(throughput_history,window=5)

plt.figure(figsize=(10,5))
plt.plot(thr_x,thr_avg)

plt.xlabel("Episode")
plt.ylabel("Average Throughput")
plt.title("Throughput vs Episode")
plt.grid(True)
plt.show()