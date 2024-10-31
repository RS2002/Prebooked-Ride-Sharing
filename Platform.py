import torch
from osrm import TSP_route
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
import numpy as np
import random

# import threading
# semaphore = threading.Semaphore(5)  # 限制同时运行的线程数为5

'''
beta_list:
beta_list[0]: reward of taking new order
beta_list[1]: reward of client paying
beta_list[2]: punishment of added time (worker salary)
beta_list[3]: add punishment to the over time
'''
def reward_func_generator(beta_list, threshold):
    def reward(time_add, direct_distance):
        if time_add <= threshold:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * time_add / 60
        else:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * time_add / 60 - beta_list[3] * (time_add - threshold) / 60
        return r
    return reward


class Platform():
    def __init__(self,discount_factor=0.99, njobs=24):
        super().__init__()
        self.reset(discount_factor)
        self.njobs = njobs

    def reset(self,discount_factor=0.99):
        self.discount_factor = discount_factor
        self.Total_Reward = 0
        self.Total_Reward_Pre = 0
        self.overtime = 0


    def assign(self,q_matrix):
        threshold = 0
        # Solve Bipartite Match Process with ILP
        num_vehicles, num_demands = q_matrix.shape
        Value_Matrix = np.concatenate((q_matrix,np.zeros_like(q_matrix)+threshold),axis=1)
        cost_matrix = -Value_Matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        # 创建一个列表来保存每个车辆被分配的订单
        assignment = [None] * len(Value_Matrix)
        # 获取每个车辆被分配的订单
        for i in range(len(row_indices)):
            if col_indices[i] >= num_demands:
                assignment[row_indices[i]] = None
            else:
                assignment[row_indices[i]] = col_indices[i]
        # 计算最大化的值
        max_value = -1 * cost_matrix[row_indices, col_indices].sum()
        # 返回分配结果和最大值
        return assignment, max_value

    def feedback(self, observe_pre, order_pre, order_pre_num, observe, assignment, new_orders_state, reward_func, punish_rate, current_time):
        feedback_table = []
        new_route_table = []
        new_route_time_table = []
        assign_state_table = []
        accepted_pre = []
        accepted_on = []

        results = Parallel(n_jobs=self.njobs)(
            delayed(excute)(observe_pre[i], order_pre[i], observe[i], assignment[i], new_orders_state, reward_func, punish_rate, current_time)
            for i
            in range(observe.shape[0]))

        # for i in range(observe.shape[0]):
        #     excute(observe_pre[i], order_pre[i], observe[i], assignment[i], new_orders_state, reward_func, punish_rate, current_time)

        for i in range(len(results)):
            result = results[i]
            reward = result[0][1]

            feedback_table.append(result[0])
            new_route_table.append(result[1])
            new_route_time_table.append(result[2])
            assign_state = result[3]
            if assign_state == 1:
                accepted_pre.append(order_pre_num[i])
                if reward[0]<0:
                    self.overtime += 1
            elif assign_state == 2:
                accepted_on.append(assignment[i])
            assign_state_table.append(assign_state)

            if reward[0] is not None:
                self.Total_Reward_Pre += reward[0] * self.discount_factor**current_time
            if reward[1] is not None:
                self.Total_Reward += reward[1] * self.discount_factor**current_time

        return feedback_table, new_route_table, new_route_time_table, accepted_pre, accepted_on, assign_state_table



def excute(observe_pre, order_pre, observe, assignment, new_orders_state, reward_func, punish_rate, current_time):
    assign_state = 0 # 0: no action, 1: pickup pre-booked order, 2: pickup on-demand order

    if observe[2]!=0: # has pre-booked order
        # 0. whether to start pick up pre-ordered order
        if observe[7]!=0: # 0.1 if the worker is not available
            # detect the time from current destination to the origin of pre-booked order
            _, _, pick_pre_time, _ = TSP_route((observe[7], observe[8]), [(observe[2], observe[3])])
            pick_pre_time = pick_pre_time[0]
            arrive_time = pick_pre_time + observe[9] + current_time # when the worker can arrive the origin of pre-booked order
            if arrive_time > observe[6]: # add overtime punishment for pre-booked order
                reward = - punish_rate # * (arrive_time - observe[6])
            else:
                reward = 0
            return [[observe_pre, order_pre, observe, None, current_time], [reward,None]], None, None, assign_state # in this circumstance, the assignment must be None
        else: # 0.2 if the worker is available
            # detect the time from current location to the origin of pre-booked order
            pick_pre_route, pick_pre_route_t, pick_pre_time, _ = TSP_route((observe[0], observe[1]), [(observe[2], observe[3])])
            pick_pre_time = pick_pre_time[0]
            arrive_time = pick_pre_time + current_time # when the worker can arrive the origin of pre-booked order
            if arrive_time > observe[6]: # add overtime punishment for pre-booked order
                reward = - punish_rate # * (arrive_time - observe[6])
            elif arrive_time == observe[6]:
                reward = 0
            if arrive_time >= observe[6]: # start to pick up pre-booked order
                if assignment is not None:
                    feedback = [[observe_pre, order_pre, observe, new_orders_state[assignment], current_time], [reward,0]]
                else:
                    feedback = [[observe_pre, order_pre, observe, None, current_time], [reward,None]]
                assign_state = 1
                return feedback, pick_pre_route, pick_pre_route_t, assign_state


    if assignment is not None: # assign on-demand order
        plat, plon, dlat, dlon = new_orders_state[assignment, :4]
        pickup_route, pickup_route_t, pickup_time, _ = TSP_route((observe[0], observe[1]), [(plat, plon)])
        pickup_time = pickup_time[0]
        work_route, work_route_t, work_time, work_distance = TSP_route((plat, plon), [(dlat, dlon)])
        work_time = work_time[0]

        # 1. detect the conflict between new assignment and pre-booked order
        if observe[2]!=0:
            _, _, pick_pre_time, _ = TSP_route((dlat, dlon), [(observe[2], observe[3])])
            pick_pre_time = pick_pre_time[0]
            arrive_time = pickup_time + work_time + pick_pre_time + current_time
            if arrive_time > observe[6]: # reject the assignment
                feedback = [[observe_pre, order_pre, observe, new_orders_state[assignment], current_time], [0, 0]]
                return feedback, None, None, assign_state


        # 2. accept the assignment
        if len(pickup_route)!=0 and len(work_route)!=0:
            pickup_route.extend(work_route)
            pickup_route_t.extend(work_route_t)
            new_route = pickup_route
            new_route_t = pickup_route_t
            new_time = pickup_time + work_time
            reward = reward_func(new_time, work_distance)
            if order_pre is None:
                feedback = [[observe_pre, None, observe, new_orders_state[assignment], current_time], [None,reward]]
            else:
                feedback = [[observe_pre, order_pre, observe, new_orders_state[assignment], current_time], [reward,reward]]
            assign_state = 2
            return feedback, new_route, new_route_t, assign_state


    feedback = [[observe_pre, order_pre, observe, None, current_time], [None,None]]
    return feedback, None, None, assign_state



