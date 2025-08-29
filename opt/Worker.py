import numpy as np
import torch
from model import Q_Net
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import pickle

INF = 1e8

def norm(order_state, worker_state, history_order_state, lat_min = 40.68878421555262, lat_max = 40.875967791801536, lon_min = -74.04528828347375, lon_max = -73.91037864632285, simulation_time = 60, max_capacity = 3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order_state, torch.Tensor):
        worker_state, history_order_state, order_state = worker_state.clone(), history_order_state.clone(), order_state.clone()
    else:
        worker_state, history_order_state, order_state = worker_state.copy(), history_order_state.copy(), order_state.copy()

    # 1. lat & lon
    order_state[:,0] = (order_state[:,0] - lat_min) / lat_range
    order_state[:,2] = (order_state[:,2] - lat_min) / lat_range
    order_state[:,1] = (order_state[:,1] - lon_min) / lon_range
    order_state[:,3] = (order_state[:,3] - lon_min) / lon_range

    worker_state[:,0] = (worker_state[:,0] - lat_min) / lat_range
    worker_state[:,1] = (worker_state[:,1] - lon_min) / lon_range

    worker_state[:,2] = (worker_state[:,2] - lat_min) / lat_range * (worker_state[:,2] != 0)
    worker_state[:,3] = (worker_state[:,3] - lon_min) / lon_range * (worker_state[:,3] != 0)
    worker_state[:,4] = (worker_state[:,4] - lat_min) / lat_range * (worker_state[:,4] != 0)
    worker_state[:,5] = (worker_state[:,5] - lon_min) / lon_range * (worker_state[:,5] != 0)

    history_order_state[:,:,0] = (history_order_state[:,:,0] - lat_min) / lat_range * (history_order_state[:,:,0] != 0)
    history_order_state[:,:,1] = (history_order_state[:,:,1] - lon_range) / lon_range * (history_order_state[:,:,1] != 0)

    # 2. time
    worker_state[:, 6] = worker_state[:, 6] / simulation_time
    worker_state[:, 9] = worker_state[:, 9] / simulation_time
    worker_state[:, 11] = worker_state[:, 11] / simulation_time
    order_state[:,4] = order_state[:,4] / simulation_time
    history_order_state[:,:,2] = history_order_state[:,:,2] / simulation_time
    history_order_state[:,:,3] = history_order_state[:,:,3] / simulation_time

    # 3. capacity
    worker_state[:, 8] = worker_state[:, 8] / max_capacity

    return order_state, worker_state, history_order_state


class Buffer():
    def __init__(self,capacity = 1e5, episode_capacity = 10):
        super().__init__()
        self.reset(capacity, episode_capacity)

    def reset(self, capacity = None, episode_capacity = None):
        if capacity is not None:
            self.capacity = capacity
            self.episode_capacity = episode_capacity

        self.num = 0

        self.worker_state = []
        self.order_state = []
        self.order_num = []

        self.action = [] # order state

        self.delta_t = []

        self.worker_state_next = []
        self.order_state_next = []
        self.order_num_next = []

        self.action_next = [] # next order state

        self.reward = []

        self.episode = []

    def append(self, experience, episode=0):
        if self.num > 0 and self.episode[0]<episode-self.episode_capacity:
            episode_np = np.array(self.episode)
            old_record_num = len(episode_np[episode_np<(episode-self.episode_capacity)])
            self.num -= old_record_num
            self.worker_state = self.worker_state[old_record_num:]
            self.order_state = self.order_state[old_record_num:]
            self.order_num = self.order_num[old_record_num:]
            self.action = self.action[old_record_num:]
            self.delta_t = self.delta_t[old_record_num:]
            self.worker_state_next = self.worker_state_next[old_record_num:]
            self.order_state_next = self.order_state_next[old_record_num:]
            self.order_num_next = self.order_num_next[old_record_num:]
            self.action_next = self.action_next[old_record_num:]
            self.reward = self.reward[old_record_num:]
            self.episode = self.episode[old_record_num:]
            if self.episode[0]<episode-self.episode_capacity:
                print("Buffer Error!")
                exit(-1)

        state, action, reward, delta_t,  state_next, action_next = experience
        if self.num == self.capacity:
            self.worker_state = self.worker_state[1:]
            self.order_state = self.order_state[1:]
            self.order_num = self.order_num[1:]
            self.action = self.action[1:]
            self.delta_t = self.delta_t[1:]
            self.worker_state_next = self.worker_state_next[1:]
            self.order_state_next = self.order_state_next[1:]
            self.order_num_next = self.order_num_next[1:]
            self.action_next = self.action_next[1:]
            self.reward = self.reward[1:]
            self.episode = self.episode[1:]
        else:
            self.num+=1

        self.worker_state.append(state[0].tolist())
        self.order_state.append(state[1].tolist())
        self.order_num.append(state[2])
        self.action.append(action.tolist())
        self.delta_t.append(delta_t)
        self.worker_state_next.append(state_next[0].tolist())
        self.order_state_next.append(state_next[1].tolist())
        self.order_num_next.append(state_next[2])
        self.action_next.append(action_next.tolist())
        self.reward.append(reward)
        # self.reward.append(reward / 10)
        # print(reward)
        self.episode.append(episode)

    def sample(self,size,device):
        if size>self.num:
            size = self.num
        if size == 0:
            return None, None, None, None, None, None, None, None, None, None

        indices = np.random.randint(0, self.num, size=size)
        # priority = np.array(self.episode)
        # priority = priority - np.min(priority) + 1
        # probabilities = np.array(priority) / np.sum(priority)
        # indices = np.random.choice(self.num, size, p=probabilities)

        worker_state = torch.tensor([self.worker_state[i] for i in indices]).to(device)
        order_state = torch.tensor([self.order_state[i] for i in indices]).to(device)
        order_num = torch.tensor([self.order_num[i] for i in indices]).to(device)
        action = torch.tensor([self.action[i] for i in indices]).to(device)
        delta_t = torch.tensor([self.delta_t[i] for i in indices]).to(device)
        worker_state_next = torch.tensor([self.worker_state_next[i] for i in indices]).to(device)
        order_state_next = torch.tensor([self.order_state_next[i] for i in indices]).to(device)
        order_num_next = torch.tensor([self.order_num_next[i] for i in indices]).to(device)
        action_next = torch.tensor([self.action_next[i] for i in indices]).to(device)
        reward = torch.tensor([self.reward[i] for i in indices]).to(device)

        return worker_state, order_state, order_num, action, delta_t, reward, worker_state_next, order_state_next, order_num_next, action_next

class Worker():
    def __init__(self, buffer, lr=0.0001, gamma=0.99, max_step=60, num=1000, device=None, zone_table_path = "./data/Manhattan_dic.pkl", model_path = None, njobs = 12, bi_direction = True, dropout = 0.0):
        super().__init__()
        self.buffer = buffer
        self.gamma = gamma
        self.device = device
        self.max_step = max_step
        self.num = num

        with open(zone_table_path, 'rb') as f:
            self.zone_dic = pickle.load(f)
        self.coordinate_lookup_lat = np.array(self.zone_dic["centroid_lat"])
        self.coordinate_lookup_lon = np.array(self.zone_dic["centroid_lon"])
        self.zone_map = np.array(self.zone_dic["map"])

        self.Q_training = Q_Net(state_size=14, history_order_size=5, current_order_size=7, hidden_dim=64, head=1, bi_direction=bi_direction, dropout=dropout).to(device)
        self.Q_target = Q_Net(state_size=14, history_order_size=5, current_order_size=7, hidden_dim=64, head=1, bi_direction=bi_direction, dropout=dropout).to(device)

        self.load(model_path,self.device)
        for param in self.Q_target.parameters():
            param.requires_grad = False
        self.Q_target.eval()
        print('Platform total parameters:', sum(p.numel() for p in self.Q_training.parameters() if p.requires_grad))
        self.update_Qtarget(tau=1.0)

        self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr) #, weight_decay=0.01)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        # self.loss_func = nn.MSELoss()
        self.loss_func = nn.MSELoss(reduction="none")
        self.njobs = njobs
        self.reset()

    def save(self, path):
        torch.save(self.Q_training.state_dict(), path)

    def load(self, path=None, device=torch.device("cpu")):
        if path is not None:
            self.Q_target.load_state_dict(torch.load(path, map_location=device))
            self.Q_training.load_state_dict(torch.load(path, map_location=device))

    def update_Qtarget(self, tau=0.005):
        for target_param, train_param in zip(self.Q_target.parameters(), self.Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def reset(self, capacity = 3, pre_rate = 0, pooling_rate = 0, train=True):
        if train:
            self.Q_training.train()
            torch.set_grad_enabled(True)
        else:
            self.Q_training.eval()
            torch.set_grad_enabled(False)
        self.is_train = train
        self.pre_rate = pre_rate
        self.pooling_rate = pooling_rate

        '''
        observation space
        0,1: current lat,lon (required to be normalized before inputting to the network, following lat and lon remain same)
        2-7: plat,plon,dlat,dlon,pre-booked time of the pre-booked order,pre-booked order type (0 allows pooling, 1 does not)
        8: remaining order place
        9: remaining picking time
        10: state -- 0 allows to pick up new orders, 1 does not (because picking up the order that doesn't allow pooling or the capacity is full)
        11: current time
        12: pre-booked rate
        13: pooling rate
        '''
        self.observe_space = np.zeros([self.num, 14])
        self.observe_space[:,8] = capacity
        self.observe_space[:,12] = self.pre_rate
        self.observe_space[:,13] = self.pooling_rate

        '''
        current orders
        0,1: drop-off lat,lon
        2: remaining transportation time (approximated)
        3: total transportation time (approximated)
        4: type (0 allows pooling, 1 does not)
        '''
        self.current_orders = np.zeros([self.num, capacity, 5])
        self.current_order_num = np.zeros([self.num])

        # allocate a initial location randomly from valid zone
        random_integers = np.random.randint(0, len(self.coordinate_lookup_lat), size=(self.num))
        self.observe_space[:, 0] = self.coordinate_lookup_lat[random_integers]
        self.observe_space[:, 1] = self.coordinate_lookup_lon[random_integers]

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = [[] for _ in range(self.num)]

        # some logs
        self.idle_time = np.zeros([self.num])
        self.max_utilization_rate = 0
        self.waiting_time_list = []
        self.waiting_time = np.zeros([self.num])
        self.Pass_Travel_Time = []
        self.start_flag = np.zeros([self.num])

    def observe(self, prebook, order, current_time, order_future = None, exploration_rate=0):
        # 0. process order state
        pid = np.array(order['PULocationID'],dtype=int)
        did = np.array(order['DOLocationID'],dtype=int)
        if len(pid) == 0:
            order = None
        else:
            pid = self.zone_map[pid-1]
            did = self.zone_map[did-1]
            minute = order['minute']
            type = order['type']
            plat, plon = self.coordinate_lookup_lat[pid], self.coordinate_lookup_lon[pid]
            dlat, dlon = self.coordinate_lookup_lat[did], self.coordinate_lookup_lon[did]
            minute = np.array(minute).reshape(-1,1)
            plat = np.array(plat).reshape(-1,1)
            plon = np.array(plon).reshape(-1,1)
            dlat = np.array(dlat).reshape(-1,1)
            dlon = np.array(dlon).reshape(-1,1)
            type = np.array(type).reshape(-1,1)
            order = np.concatenate([plat,plon,dlat,dlon,minute,type],axis=-1)

        if order_future is not None:
            pid = order_future['PULocationID']
            did = order_future['DOLocationID']
            if len(pid) != 0:
                pid = self.zone_map[pid - 1]
                did = self.zone_map[did - 1]
                minute = order_future['minute'] + 60
                type = order_future['type']
                plat, plon = self.coordinate_lookup_lat[pid], self.coordinate_lookup_lon[pid]
                dlat, dlon = self.coordinate_lookup_lat[did], self.coordinate_lookup_lon[did]
                minute = np.array(minute).reshape(-1, 1)
                plat = np.array(plat).reshape(-1, 1)
                plon = np.array(plon).reshape(-1, 1)
                dlat = np.array(dlat).reshape(-1, 1)
                dlon = np.array(dlon).reshape(-1, 1)
                type = np.array(type).reshape(-1, 1)
                order_future = np.concatenate([plat, plon, dlat, dlon, minute, type], axis=-1)
                if order is not None:
                    order = np.concatenate([order, order_future], axis = 0)
                else:
                    order = order_future

        if order is None or len(order) == 0:
            return None
        elif len(order)>self.num:
            order = order[:self.num]

        tag = np.zeros([order.shape[0],1])
        if prebook:
            tag += 1
        order = np.concatenate([order,tag],axis=-1)

        # 1. calculate q-value
        # self.observe_space[:,11] = current_time
        # x1, x2, x3 = norm(order, self.observe_space, self.current_orders)
        # x1, x2, x3 = torch.tensor(x1).to(self.device), torch.tensor(x2).to(self.device), torch.tensor(x3).to(self.device)
        # q_value = self.Q_training(x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))
        # # 2. epsilon-greedy explore
        # exploration_matrix = torch.rand_like(q_value)
        # q_value[exploration_matrix < exploration_rate] = INF
        #
        # worker_pos = self.observe_space[:, :2]
        # order_pos = order[:, :2]
        # q_value = np.sqrt(np.sum((worker_pos[:, np.newaxis, :] - order_pos[np.newaxis, :, :]) ** 2, axis=-1))
        #
        # if not prebook:
        #     # 3. delete the Q value of not available workers
        #     q_value[self.observe_space[:, 10] == 1] = -INF
        #     # 4. avoid pooling conflict
        #     for j in range(q_value.shape[1]):
        #         if order[j, -2] == 1:
        #             q_value[self.current_order_num != 0, j] = -INF
        #
        # return q_value, order
        return order

    def train(self,batch_size=512,train_times=10,show_pbar=True):
        torch.set_grad_enabled(True)
        net_train = self.Q_training
        net_target = self.Q_target

        net_train.train()
        if show_pbar:
            pbar = tqdm.tqdm(range(train_times))
        else:
            pbar = range(train_times)

        loss_list = []
        for _ in pbar:
            worker_state, order_state, order_num, action, delta_t, reward, worker_state_next, order_state_next, order_num_next, action_next = self.buffer.sample(batch_size,self.device)
            if worker_state is None:
                return -1
            x1,x2,x3 = norm(action,worker_state,order_state)
            x1_next,x2_next,x3_next = norm(action_next,worker_state_next, order_state_next)

            current_q_value = net_train(x1,x2,x3,order_num)
            current_q_value = torch.diag(current_q_value)

            next_q_value = net_target(x1_next,x2_next,x3_next,order_num_next)
            next_q_value = torch.diag(next_q_value).detach()

            is_done = (delta_t == -1).float()
            target =  reward + (self.gamma ** delta_t * next_q_value) * (1 - is_done)

            loss = self.loss_func(current_q_value.float(), target.float())
            # w = (action[:,-1]==1).float()
            # w_pre = torch.sum(w) / w.shape[0]
            # w_pre = (1 - w_pre) / (w_pre + 1e-8)
            # w[w==1] = w_pre
            # w[w==0] = 1
            # w = w.detach()
            # loss = torch.mean(loss * w)
            loss = torch.mean(loss)

            self.optim.zero_grad()
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

            self.optim.step()
            loss_list.append(loss.item())
        self.update_Qtarget()
        torch.set_grad_enabled(False)
        return np.mean(loss_list)

    def update_pre(self, assignment, order_pre):
        observe_pre = self.observe_space.copy()
        self.observe_space[:, 2:8] = 0
        order = []
        for i in range(self.num):
            if assignment[i] is None:
                order.append(None)
            else:
                order.append(order_pre[assignment[i]])
                self.observe_space[i,2:8] = order_pre[assignment[i]][:-1]
        return observe_pre, order

    def update(self, feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, assign_state_table, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.travel_route[i], self.travel_time[i], self.experience[i], feedback_table[i], new_route_table[i], new_route_time_table[i], new_remaining_time_table[i], new_total_travel_time_table[i], assign_state_table[i])
            for i in range(self.num))

        for i in range(len(results)):
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], self.travel_time[i], self.experience[i] = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5]

            if self.current_order_num[i] == 0:
                self.idle_time[i] += 1
                self.waiting_time[i] += 1

            if assign_state_table[i] != 0:
                if self.start_flag[i] == 0:
                    self.start_flag[i] = 1
                else:
                    self.waiting_time_list.append(self.waiting_time[i])
                self.waiting_time[i] = 0

            if self.is_train:
                if results[i][7] is not None:
                    self.buffer.append(results[i][7], episode)
                if results[i][6] is not None:
                    self.buffer.append(results[i][6], episode)


        if self.is_train and final_step:
            for i in range(self.num):
                if len(self.experience[i])>0:
                    self.experience[i].append(-1) # △t: -1 represents done
                    self.experience[i].append(self.experience[i][0]) # meaningless: only used to keep a same dimension
                    self.experience[i].append(self.experience[i][1])
                    self.buffer.append(self.experience[i], episode)


        utilization_rate = np.sum(self.current_order_num!=0) / self.num
        if utilization_rate > self.max_utilization_rate:
            self.max_utilization_rate = utilization_rate

def single_update(current_travel_route, current_travel_time, experience, feedback, new_route ,new_route_time ,new_remaining_time ,new_total_travel_time, assign_state):
    full_experience = None
    full_experience_pre = None

    # 1. update experience
    pickup_time = feedback[2]
    reward_list = feedback[1]
    feedback = feedback[0]
    reward_pre = reward_list[0]
    reward = reward_list[1]

    if reward_pre is not None:
        if len(experience) > 0:
            experience.append(feedback[4] - experience[0][0][11])  # △t
            experience.append(feedback[0])  # s_next
            experience.append(feedback[1])  # a_next
            full_experience_pre = experience
            experience = []
        experience.append(feedback[0])  # s_current
        experience.append(feedback[1])  # a_current
        experience.append(reward_pre)  # r

    if reward is not None:
        if len(experience) > 0:
            experience.append(feedback[4] - experience[0][0][11])  # △t
            experience.append(feedback[2])  # s_next
            experience.append(feedback[3])  # a_next
            full_experience = experience
            experience = []
        experience.append(feedback[2])  # s_current
        experience.append(feedback[3])  # a_current
        experience.append(reward)  # r

    # 2. update state
    observe_space, current_orders, current_orders_num = feedback[2]
    new_order = feedback[3]

    if assign_state == 1:  # pickup pre-booked order
        observe_space[0:2] = observe_space[2:4] # plat and plon
        observe_space[8] -= 1 # seat
        observe_space[9] = pickup_time
        observe_space[10] = 1 # not available state
        current_orders[current_orders_num,0:2] = observe_space[4:6] # dlat and dlon
        current_orders[current_orders_num,4] = observe_space[7] # order type
        current_orders_num += 1
        current_orders[:current_orders_num, 2], current_orders[:current_orders_num, 3] = new_remaining_time, new_total_travel_time
        current_travel_route, current_travel_time = new_route, new_route_time
        observe_space[2:8] = 0
    elif assign_state == 2:  # pickup on-demand order
        observe_space[0:2] = new_order[0:2]  # plat and plon
        observe_space[8] -= 1  # seat
        observe_space[9] = pickup_time
        observe_space[10] = 1  # not available state
        current_orders[current_orders_num, 0:2] = new_order[2:4]  # dlat and dlon
        current_orders[current_orders_num, 4] = new_order[5]  # order type
        current_orders_num += 1
        current_orders[:current_orders_num, 2], current_orders[:current_orders_num, 3] = new_remaining_time, new_total_travel_time
        current_travel_route, current_travel_time = new_route, new_route_time

    # 3. run 1 minute
    step = 1
    if assign_state != 0 or observe_space[9] >0:  # pick up
        if observe_space[9] > step:
            observe_space[9] -= step
            step = 0
        else:
            step -= observe_space[9]
            observe_space[9] = 0
            if current_orders[0,4] == 0 and current_orders_num < current_orders.shape[0]:
                observe_space[10] = 0  # available state

    if step > 0 and current_orders_num != 0:
        step_minute = step
        step *= 60

        for i in range(len(current_travel_time)):
            if step >= current_travel_time[i]:
                step -= current_travel_time[i]
            else:
                current_travel_time[i] -= step
                current_travel_time = current_travel_time[i:]
                current_travel_route = current_travel_route[i:]
                break
            if i == len(current_travel_time) - 1:  # finish all orders
                observe_space[0], observe_space[1] = current_travel_route[-1][1], current_travel_route[-1][0]  # lat, lon
                current_travel_time = []
                current_travel_route = []

        if len(current_travel_route) > 0:
            observe_space[0], observe_space[1] = current_travel_route[0][1], current_travel_route[0][0]  # lat, lon
        current_orders[:current_orders_num, 2] -= step_minute  # update remaining time

        # delete finished orders
        drop_index = np.zeros(current_orders.shape[0])
        drop_index[:current_orders_num] = (current_orders[:current_orders_num, 2] <= 0)
        drop_num = np.sum(drop_index)
        if drop_num > 0:
            current_orders_num -= drop_num
            observe_space[8] += drop_num
            observe_space[10] = 0  # available state
            drop_index = drop_index.astype(bool)
            finished_orders = current_orders[drop_index]
            current_orders = current_orders[~drop_index]
            fill_matrix = np.zeros_like(finished_orders)
            current_orders = np.concatenate([current_orders, fill_matrix], axis=0)

    return observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, experience, full_experience, full_experience_pre


