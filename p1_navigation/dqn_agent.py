import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
EPS = 1e-6              # define a very small value
ALPHA = 0.6             # hyperparam for prioritized experience replay - prob of transitions
BETA = 0.4              # hyperparam for prioritized experience replay - importance-sampling 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBufferWithPriority(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences_with_index = self.memory.sample()
                self.learn(experiences_with_index, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences_with_index, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences_with_index (Tuple[torch.Variable]): tuple of (s, a, r, s', done, index, weightsIS) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, index, weightsIS = experiences_with_index

        ## TODO: compute and minimize the loss

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local.forward(states).gather(1, actions)
        
        # Compute importance-sampling weight
        
        # Compute loss
        loss_fn = nn.MSELoss(reduce=False)
        loss = loss_fn(Q_expected, Q_targets)
        weighted_loss = torch.mean(torch.from_numpy(weightsIS).float() * loss) 
        # Update priority according to TD error
        self.memory.update_priority(list(loss.detach().numpy().squeeze()**ALPHA+EPS), index)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBufferWithPriority:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.priority = []
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priority.append(1 if len(self.priority)==0 else max(self.priority))
        assert len(self.memory) == len(self.priority), "memory size is not equal to priority size"
    
    def sample(self):
        """Randomly sample a batch of experiences, its index and importance-sampling weights from memory by priority."""
        probs = np.array([p/sum(self.priority) for p in self.priority], dtype='float').round(5) # truncate the precision
        probs[-1] = 1.0 - sum(probs[:-1])
        assert sum(probs)==1, "probs is not sum up to 1 as {}".format(sum(probs)) # numpy issue- floatvalue with high precisionnot not sum to 1
        
        index = np.random.choice(range(self.__len__()), size=self.batch_size, p=probs)
        
        weightsIS   = [(self.__len__()*probs[i])**(-BETA) for i in index]
        weightsIS   = np.array([w / max(weightsIS) for w in weightsIS]).reshape((-1, 1)) # normalize by max
        states = torch.from_numpy(np.vstack([self.memory[i].state for i in index])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in index])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in index])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in index])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in index]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, index, weightsIS)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_priority(self, new_priority, index):
        for i, p in zip(index, new_priority):
            self.priority[i] = p
        