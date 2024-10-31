import math
import numpy as np
import torch
import pandas as pd

from model import Q_Net
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import pickle

INF = 1e8

class Buffer():
    def __init__(self,capacity = 1e5):
        super().__init__()
        self.reset(capacity)

    def reset(self, capacity = None):
        if capacity is not None:
            self.capacity = capacity

        self.num = 0

        self.state = [] # worker state
        self.action = [] # order state

        self.delta_t = []

        self.state_next = [] # next worker state
        self.action_next = [] # next order state

        self.reward = []

        self.episode = []

    def append(self, experience, episode=0):
        state, action, delta_t, reward, state_next, action_next = experience
        if self.num == self.capacity:
            self.state = self.state[1:]
            self.action = self.action[1:]
            self.delta_t = self.delta_t[1:]
            self.state_next = self.state_next[1:]
            self.action_next = self.action_next[1:]
            self.reward = self.reward[1:]
            self.episode = self.episode[1:]
        else:
            self.num+=1
        self.state.append(state.tolist())
        self.action.append(action.tolist())
        self.delta_t.append(delta_t)
        self.state_next.append(state_next.tolist())
        self.action_next.append(action_next.tolist())
        self.reward.append(reward)
        self.episode.append(episode)

    def sample(self,size,device):
        # indices = np.random.randint(0, self.num, size=size)
        priority = np.array(self.episode)
        priority = priority - np.min(priority) + 1
        probabilities = np.array(priority) / np.sum(priority)
        indices = np.random.choice(self.num, size, p=probabilities)

        state = torch.tensor([self.state[i] for i in indices]).to(device)
        action = torch.tensor([self.action[i] for i in indices]).to(device)
        delta_t = torch.tensor([self.delta_t[i] for i in indices]).to(device)
        state_next = torch.tensor([self.state_next[i] for i in indices]).to(device)
        action_next = torch.tensor([self.action_next[i] for i in indices]).to(device)
        reward = torch.tensor([self.reward[i] for i in indices]).to(device)

        return state, action, delta_t, reward,  state_next, action_next



def norm(worker_state, order_state, lat_min = 40.68878421555262, lat_max = 40.875967791801536, lon_min = -74.04528828347375, lon_max = -73.91037864632285, simulation_time = 60):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order_state, torch.Tensor):
        worker_state, order_state = worker_state.clone(), order_state.clone()
    else:
        worker_state, order_state = worker_state.copy(), order_state.copy()

    # 1. lat & lon
    order_state[:,0] = (order_state[:,0] - lat_min) / lat_range
    order_state[:,2] = (order_state[:,2] - lat_min) / lat_range
    order_state[:,1] = (order_state[:,1] - lon_min) / lon_range
    order_state[:,3] = (order_state[:,3] - lon_min) / lon_range

    worker_state[:,0] = (worker_state[:,0] - lat_min) / lat_range
    worker_state[:,1] = (worker_state[:,1] - lon_min) / lon_range

    worker_state[:,2] = (worker_state[:,2] - lat_min) / lat_range
    worker_state[:,3] = (worker_state[:,3] - lon_min) / lon_range
    worker_state[:,4] = (worker_state[:,4] - lat_min) / lat_range
    worker_state[:,5] = (worker_state[:,5] - lon_min) / lon_range
    worker_state[:,7] = (worker_state[:,7] - lat_min) / lat_range
    worker_state[:,8] = (worker_state[:,8] - lon_min) / lon_range

    # 2. time
    worker_state[:, 6] = worker_state[:, 6] / simulation_time
    worker_state[:, 9] = worker_state[:, 9] / simulation_time
    worker_state[:, 10] = worker_state[:, 10] / simulation_time
    order_state[:,4] = order_state[:,4] / simulation_time

    return worker_state, order_state

class Worker():
    def __init__(self, buffer, buffer_pre, lr=0.0001, gamma=0.99, max_step=60, num=1000, device=None, zone_table_path = "./data/Manhattan_dic.pkl", model_path = None, model_pre_path = None, njobs = 24, arl = True, dropout = 0.0):
        super().__init__()
        self.buffer = buffer
        self.buffer_pre = buffer_pre

        self.gamma = gamma
        self.device = device
        self.max_step = max_step
        self.num = num

        with open(zone_table_path, 'rb') as f:
            self.zone_dic = pickle.load(f)
        # self.zone_lookup = self.zone_dic["zone_num"]
        self.coordinate_lookup_lat = np.array(self.zone_dic["centroid_lat"])
        self.coordinate_lookup_lon = np.array(self.zone_dic["centroid_lon"])
        self.zone_map = np.array(self.zone_dic["map"])

        self.Q_training = Q_Net(worker_state_size = 11, order_state_size = 5, hidden_dim=64, head=1, arl=arl, dropout=dropout).to(device)
        self.Q_target = Q_Net(worker_state_size = 11, order_state_size = 5, hidden_dim=64, head=1, arl=arl, dropout=dropout).to(device)
        self.Q_training_pre = Q_Net(worker_state_size = 11, order_state_size = 5, hidden_dim=64, head=1, arl=arl, dropout=dropout).to(device)
        self.Q_target_pre = Q_Net(worker_state_size = 11, order_state_size = 5, hidden_dim=64, head=1, arl=arl, dropout=dropout).to(device)


        self.load(model_path,model_pre_path,self.device)
        for param in self.Q_target.parameters():
            param.requires_grad = False
        self.Q_target.eval()
        for param in self.Q_target_pre.parameters():
            param.requires_grad = False
        self.Q_target_pre.eval()
        print('Platform total parameters:', 2 * sum(p.numel() for p in self.Q_training.parameters() if p.requires_grad))
        self.update_Qtarget(tau=0.0)

        self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr, weight_decay=0.0)
        self.optim_pre = torch.optim.Adam(self.Q_training_pre.parameters(), lr=lr, weight_decay=0.0)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        self.schedule_pre = torch.optim.lr_scheduler.ExponentialLR(self.optim_pre, gamma=0.99)

        self.loss_func = nn.MSELoss()

        self.njobs = njobs

        self.reset()

    def reset(self,train=True):
        if train:
            self.Q_training.train()
            self.Q_training_pre.train()
            torch.set_grad_enabled(True)
        else:
            self.Q_training.eval()
            self.Q_training_pre.eval()
            torch.set_grad_enabled(False)

        self.is_train = train

        '''
        observation space
        0,1: current lat,lon (required to be normalized before inputting to the network, following lat and lon remain same)
        2-6: plat,plon,dlat,dlon,pre-booked time of the pre-booked order
        7-9: dlat,dlon,remaining time of picked order
        10: current time
        '''
        self.observe_space = np.zeros([self.num, 11])

        # allocate a initial location randomly from valid zone
        random_integers = np.random.randint(0, len(self.coordinate_lookup_lat), size=(self.num))
        self.observe_space[:, 0] = self.coordinate_lookup_lat[random_integers]
        self.observe_space[:, 1] = self.coordinate_lookup_lon[random_integers]

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = [[] for _ in range(self.num)]
        self.experience_pre = [[] for _ in range(self.num)]

    def save(self, path1, path2):
        torch.save(self.Q_training.state_dict(), path1)
        torch.save(self.Q_training_pre.state_dict(), path2)

    def load(self, path1 = None, path2 = None, device = torch.device("cpu")):
        if device == torch.device("cpu"):
            if path1 is not None:
                self.Q_target.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
                self.Q_training.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
            if path2 is not None:
                self.Q_training_pre.load_state_dict(torch.load(path2,map_location=torch.device('cpu')))
                self.Q_target_pre.load_state_dict(torch.load(path2,map_location=torch.device('cpu')))
        else:
            if path1 is not None:
                self.Q_target.load_state_dict(torch.load(path1))
                self.Q_training.load_state_dict(torch.load(path1))
            if path2 is not None:
                self.Q_training_pre.load_state_dict(torch.load(path2))
                self.Q_target_pre.load_state_dict(torch.load(path2))

    def update_Qtarget(self,tau=0.005):
        for target_param, train_param in zip(self.Q_target.parameters(), self.Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)
        for target_param, train_param in zip(self.Q_target_pre.parameters(), self.Q_training_pre.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def observe(self, network, order, current_time, exploration_rate=0):
        # 0. process order state
        pid = order['PULocationID']
        did = order['DOLocationID']
        pid = self.zone_map[pid-1]
        did = self.zone_map[did-1]
        minute = order['minute']
        plat, plon = self.coordinate_lookup_lat[pid], self.coordinate_lookup_lon[pid]
        dlat, dlon = self.coordinate_lookup_lat[did], self.coordinate_lookup_lon[did]
        minute = np.array(minute).reshape(-1,1)
        plat = np.array(plat).reshape(-1,1)
        plon = np.array(plon).reshape(-1,1)
        dlat = np.array(dlat).reshape(-1,1)
        dlon = np.array(dlon).reshape(-1,1)
        order = np.concatenate([plat,plon,dlat,dlon,minute],axis=-1)

        torch.set_grad_enabled(False)
        # 1. calculate q-value
        self.observe_space[:,-1] = current_time
        worker_state, order_state = norm(self.observe_space, order)
        worker_state, order_state = torch.tensor(worker_state).to(self.device), torch.tensor(order_state).to(self.device)
        q_value = network(worker_state, order_state)
        # 2. epsilon-greedy explore
        exploration_matrix = torch.rand_like(q_value)
        q_value[exploration_matrix < exploration_rate] = INF
        # 3. delete the Q value of not available workers
        q_value[self.observe_space[:,7]!=0] = -INF
        return q_value.cpu().detach().numpy(), order

    def train(self,buffer,net_train,net_target,optim,schedule,batch_size=512,train_times=10):
        torch.set_grad_enabled(True)
        pbar = tqdm.tqdm(range(train_times))
        loss_list = []
        for _ in pbar:
            state, action, delta_t, reward, state_next, action_next = buffer.sample(batch_size,self.device)
            state, action = norm(state, action)
            state_next, action_next = norm(state_next, action_next)

            current_q_value = net_train(state, action)
            current_q_value = torch.diag(current_q_value)

            next_q_value1 = net_train(state_next, action_next)
            next_q_value1 = torch.diag(next_q_value1).detach()
            next_q_value2 = net_target(state_next, action_next)
            next_q_value2 = torch.diag(next_q_value2).detach()
            next_q_value = torch.min(next_q_value1,next_q_value2)

            is_done = (delta_t == -1).float()
            target =  reward + (self.gamma ** delta_t * next_q_value) * (1 - is_done)

            loss = self.loss_func(current_q_value.float(),target.float())
            optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net_train.parameters(), 1.0)  # avoid gradient explosion
            has_nan = False
            for name, param in net_train.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break
            if has_nan:
                continue

            optim.step()
            loss_list.append(loss.item())
        schedule.step()
        return np.mean(loss_list)

    def update_pre(self, assignment, order_pre):
        observe_pre = self.observe_space.copy()
        self.observe_space[:, 2:7] = 0
        order = []
        for i in range(self.num):
            if assignment[i] is None:
                order.append(None)
            else:
                order.append(order_pre[assignment[i]])
                self.observe_space[i,2:7] = order_pre[assignment[i]]
        return observe_pre, order


    def update(self, feedback_table, new_route_table ,new_route_time_table, assign_state_table, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.travel_route[i], self.travel_time[i], self.experience[i], self.experience_pre[i], feedback_table[i], new_route_table[i], new_route_time_table[i], assign_state_table[i])
            for i in range(self.num))
        # for i in range(self.num):
        #     single_update(self.travel_route[i], self.travel_time[i], self.experience[i], self.experience_pre[i], feedback_table[i], new_route_table[i], new_route_time_table[i], assign_state_table[i])

        for i in range(len(results)):
            self.observe_space[i], self.travel_route[i], self.travel_time[i], self.experience[i], self.experience_pre[i] = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]

            if self.is_train:
                if results[i][5] is not None:
                    self.buffer.append(results[i][5], episode)
                if results[i][6] is not None:
                    self.buffer_pre.append(results[i][6], episode)
        if final_step:
            for i in range(self.num):
                if len(self.experience[i])>0:
                    self.experience[i].append(-1) # △t: -1 represents done
                    self.experience[i].append(self.experience[i][0]) # meaningless: only used to keep a same dimension
                    self.experience[i].append(self.experience[i][1])
                    self.buffer.append(self.experience[i], episode)
                if len(self.experience_pre[i])>0:
                    self.experience_pre[i].append(-1) # △t: -1 represents done
                    self.experience_pre[i].append(self.experience_pre[i][0]) # meaningless: only used to keep a same dimension
                    self.experience_pre[i].append(self.experience_pre[i][1])
                    self.buffer.append(self.experience_pre[i], episode)


def single_update(current_travel_route, current_travel_time, experience, experience_pre, feedback, new_route, new_route_time, assign_state):
    full_experience = None
    full_experience_pre = None

    # 1. update experience
    reward_list = feedback[1]
    feedback = feedback[0]
    reward_pre = reward_list[0]
    reward = reward_list[1]

    if reward_pre is not None:
        if len(experience_pre) > 0:
            experience_pre.append(feedback[0][-1] - experience_pre[0][-1])  # △t
            experience_pre.append(feedback[0])  # s_next
            experience_pre.append(feedback[1])  # a_next
            full_experience_pre = experience_pre
            experience_pre = []
        experience_pre.append(feedback[0])  # s_current
        experience_pre.append(feedback[1])  # a_current
        experience_pre.append(reward_pre)  # r

    if reward is not None:
        if len(experience) > 0:
            experience.append(feedback[2][-1] - experience[0][-1])  # △t
            experience.append(feedback[2])  # s_next
            experience.append(feedback[3])  # a_next
            full_experience = experience
            experience = []
        experience.append(feedback[2])  # s_current
        experience.append(feedback[3])  # a_current
        experience.append(reward)  # r

    # 2. update state
    observe_space = feedback[2]
    new_order = feedback[3]

    if assign_state == 1: # pickup pre-booked order
        observe_space[7:9] = observe_space[4:6]
        observe_space[9] = np.sum(new_route_time)
        observe_space[2:7] = 0
        current_travel_route, current_travel_time = new_route, new_route_time
    elif assign_state == 2: # pickup on-demand order
        observe_space[7:9] = new_order[2:4]
        observe_space[9] = np.sum(new_route_time)
        current_travel_route, current_travel_time = new_route, new_route_time

    # 3. run 1 minute
    if len(current_travel_time)!=0:
        observe_space[9] -= 1
        if observe_space[9]<=0: # order finish
            current_travel_route, current_travel_time = [], []
            observe_space[7:10] = 0
        else:
            step = 60 # 60 second
            for i in range(len(current_travel_time)):
                if step >= current_travel_time[i]:
                    step -= current_travel_time[i]
                else:
                    current_travel_time[i] -= step
                    current_travel_time = current_travel_time[i:]
                    current_travel_route = current_travel_route[i:]
                    break
    return observe_space, current_travel_route, current_travel_time, experience, experience_pre, full_experience, full_experience_pre
