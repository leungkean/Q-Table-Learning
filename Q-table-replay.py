import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from copy import deepcopy
import argparse

from afa.data import load_unsupervised_split_as_numpy, load_supervised_split_as_numpy
from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.environments.core import DirectClassificationEnv

import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle
import wandb
import ray

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='molecule_20',  help='Dataset used for environment')
parser.add_argument('--lr', type=float, default=1.0,          help='Learning rate')
parser.add_argument('--y', type=float, default=0.99,           help='Discount factor (gamma)')
parser.add_argument('--eps', type=float, default=1.0,          help='Epsilon for epsilon-greedy policy')
parser.add_argument('--eps_decay', type=float, default=1-5e-5, help='Epsilon decay')
parser.add_argument('--min_eps', type=float, default=1e-4,     help='Minimum epsilon')
parser.add_argument('--cost', type=float, default=0.01,        help='Acquisition cost')

args = parser.parse_args()

config = {"env": args.env, "lr": args.lr, "y": args.y, "eps": args.eps, "eps_decay": args.eps_decay, "min_eps": args.min_eps, "cost": args.cost}
wandb.init(name='Q-table', project="deep-rl-tf2", config=config)

class ReplayBuffer:
    def __init__(self, capacity=10000, batch_size=512):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)

    def shuffle(self):
        random.shuffle(self.buffer)

class QTable:
    def __init__(self, env, lr=args.lr, y=args.y, eps=args.eps, eps_decay=args.eps_decay, min_eps=args.min_eps, train_size=5241):
        self.env = env
        self.state_dim = env.observation_space['observed'].shape[0]
        self.action_dim = env.action_space.n

        # Set learning parameters
        self.lr = lr
        self.y = y
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.train_size = train_size
        self.num_episodes = 50*self.train_size

        # Keep list of states used to build the Q-table 
        # Starts off with just the initial state
        self.q_table = {} 

        # Keep list of states and actions used to build the Q-table
        self.visit = {}

        # Initialize Replay
        self.replay_buffer = ReplayBuffer()

    def choose_action(self, q_row, actions_remain, state):
        valid_actions = deepcopy(actions_remain)
        # Choose action based on the Q-table 
        if np.random.random() < self.eps or np.count_nonzero(q_row == -1) == q_row.shape[0]:
            np.random.shuffle(valid_actions)
            return valid_actions[0]
        else:
            best_action = np.flip(np.argsort(q_row))
            for a in best_action:
                if a in valid_actions:
                    return a

    def replay(self, iters=10):
        if self.replay_buffer.size() <= self.replay_buffer.batch_size: return
        for _ in range(iters):
            samples = self.replay_buffer.sample()
            for sample in samples:
                state, action, reward, next_state, done = sample
                if not done:
                    self.q_table[state][action] += (self.lr/self.visit[(state, action)]) * (reward + self.y * np.max(self.q_table[next_state]) - self.q_table[state][action])
                    self.visit[(state, action)] += 1
                else:
                    self.q_table[state][action] += (self.lr/self.visit[(state, action)]) * (reward - self.q_table[state][action])
                    self.visit[(state, action)] += 1

    def train(self): 
        #create lists to contain total rewards and accuracy
        rList = []
        acc = []
        feat_length = []

        new_states = 0 
        reused_states = 0

        for ep in tqdm(range(self.num_episodes)):
            s = tuple(self.env.reset()['observed'].astype(np.byte)) 
            d = False 

            total_reward = 0 
            actions_remain = list(range(20, self.action_dim))

            while True:
                # Initialize Q-table row for new state s
                if s not in self.q_table:
                    self.q_table[s] = np.zeros(self.action_dim) - 1.0 
                    new_states += 1
                # Choose an action by greedily (with e chance of random action) from the Q-table 
                action = self.choose_action(self.q_table[s], actions_remain, s)
                actions_remain.remove(action)
                if (s,action) not in self.visit:
                    self.visit[(s,action)] = 1
                else:
                    self.visit[(s,action)] += 1
                # Get new state and reward from environment 
                s1, r, d, info = self.env.step(action) 
                s1 = tuple(s1['observed'].astype(np.byte)) 
                self.replay_buffer.put(s, action, r, s1, d) 
                total_reward += r
                # Initialize Q-table row for new state s1 
                if s1 not in self.q_table: 
                    self.q_table[s1] = np.zeros(self.action_dim) - 1.0 
                    new_states += 1
                else:
                    reused_states += 1
                # Update Q-table for current state s and action a 
                if not d:
                    self.q_table[s][action] += (self.lr/self.visit[(s,action)])*(r + self.y*np.max(self.q_table[s1]) - self.q_table[s][action]) 
                else: 
                    #If done then update q_value without future rewards 
                    self.q_table[s][action] += (self.lr/self.visit[(s,action)])*(r - self.q_table[s][action]) 
                    feat_length.append(np.count_nonzero(list(s))) 
                    acc.append(1 if action-self.state_dim == info['target'] else 0) 
                    rList.append(total_reward) 
                    break

                s = deepcopy(s1)

            # Replay past experiences
            self.replay()

            # Print info
            if ep % 100 == 0 and ep > 0: 
                self.replay_buffer.shuffle()
                acc_mean = np.mean(acc) 
                rList_mean = np.mean(rList) 
                print("Episode: {} Mean Reward: {}".format(ep, rList_mean)) 
                print("Accuracy: {}".format(acc_mean)) 
                print("New States: {} ({}%)".format(new_states, new_states*100/(new_states+reused_states)))
                print("Q-table size: {}".format(len(self.q_table)))
                wandb.log({'Reward': rList_mean, 'Accuracy': acc_mean, 'New States(%):': new_states*100/(new_states+reused_states), 'Number of Features Used': np.mean(feat_length)})
                acc = []
                rList = []
                feat_length = []
                new_states = 0
                reused_states = 0

            # Decay epsilon 
            self.eps *= self.eps_decay 
            self.eps = max(self.eps, self.min_eps) 

        # Save Q-table
        with open(f"q_table_{args.cost}.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
            
    def test(self, test_env, test_size=640): 
        acc = [] 
        feat_length = [] 

        for ep in tqdm(range(test_size)):
            s = tuple(test_env.reset()['observed'].astype(np.byte)) 
            d = False 

            total_reward = 0 
            next_action = np.argmax(self.q_table[s]) 
            actions_remain = list(range(20, self.action_dim))
            actions_remain.remove(next_action)

            while not d: 
                # Choose action based on argmax of Q-table 
                action = next_action 
                # Get new state and reward from environment 
                s, r, d, info = test_env.step(action) 
                s = tuple(s['observed'].astype(np.byte)) 
                total_reward += r 
                # Get next action 
                if s not in self.q_table: 
                    next_action = np.random.choice(actions_remain)
                    actions_remain.remove(next_action)
                else: 
                    best_action = np.flip(np.argsort(self.q_table[s]))
                    for a in best_action: 
                        if a in actions_remain:
                            next_action = a
                            actions_remain.remove(next_action)
                            break

            acc.append(1 if action-self.state_dim == info['target'] else 0) 
            feat_length.append(np.count_nonzero(list(s)))

        print("Test Accuracy: {}".format(np.mean(acc)))
        print("Test Number of Features Used: {}".format(np.mean(feat_length)))
        wandb.log({'Test Accuracy': np.mean(acc), 'Test Number of Features Used': np.mean(feat_length)})


def main(): 
    features, targets = load_supervised_split_as_numpy(args.env, 'train')
    dataset_manager = EnvironmentDatasetManager.remote( 
            features, targets
    )

    env = DirectClassificationEnv(
            dataset_manager,
            incorrect_reward=-1.0,
            acquisition_cost=np.array([0.0 for _ in range(20)] + [args.cost for _ in range(20)]).astype(np.float32),
    )

    qtable = QTable(env)
    print(qtable.action_dim)
    qtable.train()

    test_feat, test_targets = load_supervised_split_as_numpy(args.env, 'test')
    test_dataset_manager = EnvironmentDatasetManager.remote(
            test_feat, test_targets
    )
    test_env = DirectClassificationEnv(
            test_dataset_manager,
            incorrect_reward=-1.0,
            acquisition_cost=np.array([0.0 for _ in range(20)] + [args.cost for _ in range(20)]).astype(np.float32),
    )
    qtable.test(test_env)

if __name__ == "__main__":
    main()
