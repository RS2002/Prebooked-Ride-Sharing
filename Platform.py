from osrm import TSP_route
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
import numpy as np

def assign(q_matrix,pad=True):
    threshold = 0
    # Solve Bipartite Match Process with ILP
    num_vehicles, num_demands = q_matrix.shape
    if pad:
        Value_Matrix = np.concatenate((q_matrix,np.zeros_like(q_matrix)+threshold),axis=1)
    else:
        Value_Matrix = q_matrix
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


'''
beta_list:
beta_list[0]: reward of taking new order
beta_list[1]: reward of client paying (proportion to distance)
beta_list[2]: punishment of picking up time
beta_list[3]: punishment of timeout orders
beta_list[4]: punishment of added time
'''
def reward_func_generator(beta_list, threshold):
    def reward(time_add,time_out,pickup_time,direct_distance):
        if time_add <= threshold:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * pickup_time / 60 - beta_list[3] * time_out - beta_list[4] * time_add / 60
        else:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * pickup_time / 60 - beta_list[3] * time_out - beta_list[4] * time_add / 60 - beta_list[5] * (time_add-threshold) / 60
        return r
    return reward

class Platform():
    def __init__(self,discount_factor=0.99, njobs=24):
        super().__init__()
        self.reset(discount_factor)
        self.njobs = njobs

    def reset(self,discount_factor=0.99):
        self.discount_factor = discount_factor
        self.discount_factor_pre = discount_factor


        self.Total_Reward = 0
        self.Total_Reward_Pre = 0

        self.Pickup_Num = [0]*4 # total picking up order numbers of four types
        self.Pickup_Time = [0]*4
        self.Waiting_Time = [0]*4
        self.Confirmation_Time = [0]*2
        self.Detour = [0]*4
        self.Overtime_num = [0]*2 # total overtime pre-booked order amounts of two types
        self.Overtime = [0]*2 # total overtime of two types
        self.Reward = [0]*4 # total reward of each type
        self.Swap = 0 # total swap times of pre-booked orders
        self.Assigment = 0 # total assignment times of pre-booked orders


    def feedback(self, observe_pre, order_pre, assignment_pre, observe, current_order_state, current_order_num, assignment, new_orders_state, time_threshold, reward_func, reward_parameter_list, current_time):
        feedback_table = []
        new_route_table = []
        new_route_time_table = []
        new_remaining_time_table = []
        new_total_travel_time_table = []
        assign_state_table = []
        accepted_pre = [] # number of picked pre-booked orders
        accepted_on = [] # number of picked on-demand orders

        results = Parallel(n_jobs=self.njobs)(
            delayed(excute)(observe_pre[i], order_pre[i], observe[i], current_order_state[i], current_order_num[i], assignment[i], new_orders_state, time_threshold, reward_func, reward_parameter_list, current_time)
            for i
            in range(observe.shape[0]))

        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            new_route_table.append(result[1])
            new_route_time_table.append(result[2])
            new_remaining_time_table.append(result[3])
            new_total_travel_time_table.append(result[4])
            assign_state_table.append(result[5])

            log = result[6]

            swap = log["swap"]
            if swap != -1:
                self.Assigment += 1
                self.Swap += swap

            assign_state = result[5]
            if assign_state == 1:
                accepted_pre.append(assignment_pre[i])
                overtime = log["wait"]
                pickup_time = log["pickup"]
                overtime_num = int(overtime>0)
                workload = log["workload"]
                direct = log["direct"]
                r = log["reward"]
                waiting_time = log["wait"]
                detour = log["detour"]
                if order_pre[i][5] == 0: # pooling
                    self.Pickup_Num[0] += 1
                    self.Pickup_Time[0] += pickup_time
                    self.Detour[0] += detour
                    self.Overtime[0] += overtime
                    self.Overtime_num[0] += overtime_num
                    self.Reward[0] += r
                    self.Waiting_Time[0] += waiting_time
                elif order_pre[i][5] == 1: # non pooling
                    self.Pickup_Num[1] += 1
                    self.Pickup_Time[1] += pickup_time
                    self.Detour[1] += detour
                    self.Overtime[1] += overtime
                    self.Overtime_num[1] += overtime_num
                    self.Reward[1] += r
                    self.Waiting_Time[1] += waiting_time
                else:
                    print("Pre-booked Error")
                    exit(-1)

            elif assign_state == 2:
                accepted_on.append(assignment[i])
                pickup_time = log["pickup"]
                workload = log["workload"]
                direct = log["direct"]
                r = log["reward"]
                waiting_time = log["wait"]
                detour = log["detour"]
                confirmation_time = log["confirmation"]
                if new_orders_state[assignment[i]][5] == 0: # pooling
                    self.Pickup_Num[2] += 1
                    self.Pickup_Time[2] += pickup_time
                    self.Detour[2] += detour
                    self.Reward[2] += r
                    self.Waiting_Time[2] += waiting_time
                    self.Confirmation_Time[0] += confirmation_time
                elif new_orders_state[assignment[i]][5] == 1: # non pooling
                    self.Pickup_Num[3] += 1
                    self.Pickup_Time[3] += pickup_time
                    self.Detour[3] += detour
                    self.Reward[3] += r
                    self.Waiting_Time[3] += waiting_time
                    self.Confirmation_Time[1] += confirmation_time
                else:
                    print("On-demand Error")
                    exit(-1)

            reward_pre, reward = result[0][1]
            if reward_pre is not None:
                self.Total_Reward_Pre += reward_pre * self.discount_factor_pre**current_time
            if reward is not None:
                self.Total_Reward += reward * self.discount_factor**current_time

        return feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, assign_state_table, accepted_pre, accepted_on

'''
reward_parameter_list: 
0 -- on-time reward
1 -- conflict punishment
2 -- punishment scale 1 (used for those have been already over time)
3 -- conflict punishment (for on-demand agent)
4 -- indirect reward rate (for pre-booked agent)
5 -- punishment scale 2 (for real over time)
6 -- pickup reward
'''
def excute(observe_pre, order_pre, observe, current_order_state, current_order_num, assignment, new_orders_state, time_threshold, reward_func, reward_parameter_list, current_time):
    assign_state = 0 # 0: no action, 1: pick up pre-booked order, 2: pick up on-demand order
    reward_pre, reward = None, None
    current_order_num = int(current_order_num)

    worker_type = observe[10]
    rest_picking_time = observe[9]
    curr_lat, curr_lon = observe[0], observe[1]

    log = {}
    log["swap"] = -1

    # 0. whether to start pick up pre-booked order
    if order_pre is not None: # has pre-booked order
        order_pre_type = observe[7]
        if (observe[2:8] == observe_pre[2:8]).all():
            log["swap"] = 0
        else:
            log["swap"] = 1

        if worker_type == 1: # if the worker is not available (current_order_num>=1)
            if order_pre_type == 0: # if the pre-booked order allows pooling
                if current_order_state[0,4] == 0: # if the unfinished orders allow pooling
                    if current_order_num == current_order_state.shape[0]: # need to wait until having a new available seat
                        first_order_index = np.argmin(current_order_state[:, 2])
                        rest_finishing_time = current_order_state[first_order_index, 2]
                        dest_lat, dest_lon = current_order_state[first_order_index, 0], current_order_state[first_order_index, 1]
                        _, _, pick_pre_time, _ = TSP_route((dest_lat, dest_lon), [(observe[2], observe[3])])
                        pick_pre_time = pick_pre_time[0]
                        arrive_time = pick_pre_time + rest_finishing_time + rest_picking_time + current_time  # when the worker can arrive the origin of pre-booked order
                    else: # need to wait until finishing current picking up
                        _, _, pick_pre_time, _ = TSP_route((curr_lat, curr_lon), [(observe[2], observe[3])])
                        pick_pre_time = pick_pre_time[0]
                        arrive_time = pick_pre_time + rest_picking_time + current_time
                else: # if the unfinished orders does not allow pooling
                    # need to wait until the picked order finishes
                    rest_finishing_time = current_order_state[0, 2]
                    dest_lat, dest_lon = current_order_state[0, 0], current_order_state[0, 1]
                    _, _, pick_pre_time, _ = TSP_route((dest_lat, dest_lon), [(observe[2], observe[3])])
                    pick_pre_time = pick_pre_time[0]
                    arrive_time = pick_pre_time + rest_finishing_time + rest_picking_time + current_time
            else: # if the pre-booked order does not allow pooling
                # need to wait until all orders finish
                last_order_index = np.argmax(current_order_state[:, 2])
                rest_finishing_time = current_order_state[last_order_index, 2]
                dest_lat, dest_lon = current_order_state[last_order_index, 0], current_order_state[last_order_index, 1]
                _, _, pick_pre_time, _ = TSP_route((dest_lat, dest_lon), [(observe[2], observe[3])])
                pick_pre_time = pick_pre_time[0]
                arrive_time = pick_pre_time + rest_finishing_time + rest_picking_time + current_time

            if arrive_time > observe[6]:  # add overtime punishment for pre-booked order
                reward_pre = - reward_parameter_list[1] * (arrive_time - observe[6])
                if current_time > observe[6]:
                    reward_pre *= reward_parameter_list[2]
            else:
                reward_pre = 0
            return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], None, current_time], [reward_pre, reward], None], None, None, None, None, assign_state, log  # in this circumstance, the assignment must be None
        else: # if the worker is available
            if order_pre_type == 0: # if the pre-booked order allows pooling
                _, _, pick_pre_time, _ = TSP_route((curr_lat, curr_lon), [(observe[2], observe[3])])
                pick_pre_time = pick_pre_time[0]
                arrive_time = pick_pre_time + current_time
                if arrive_time >= observe[6]: # start to pick up the pre-booked order
                    assign_state = 1
                    if arrive_time > observe[6]:
                        reward_pre = - reward_parameter_list[1] * (arrive_time - observe[6])
                        reward_pre *= reward_parameter_list[5]
                        # if current_time > observe[6]:
                        #     reward_pre *= reward_parameter_list[2]
                    elif arrive_time == observe[6]:
                        reward_pre = reward_parameter_list[0]

                    destination_points = []
                    for i in range(current_order_num):
                        destination_points.append((current_order_state[i, 0], current_order_state[i, 1]))
                    destination_points.append((observe[4], observe[5]))
                    new_route, new_route_time, new_time, _ = TSP_route((observe[2], observe[3]), destination_points)
                    new_total_travel_time = np.array(new_time)
                    new_total_travel_time[:-1] = new_total_travel_time[:-1] + current_order_state[:current_order_num, 3] - current_order_state[:current_order_num, 2]  # add the time already cost for each old order
                    new_total_travel_time[:-1] = new_total_travel_time[:-1] + pick_pre_time
                    pick_pre_time2 = arrive_time - observe[6] # the waiting time of pre-booked customer
                    new_total_travel_time[-1] = new_total_travel_time[-1] + pick_pre_time2

                    _, _, direct_time, direct_distance = TSP_route((observe[2], observe[3]), [(observe[4], observe[5])])
                    timeout = np.sum(new_total_travel_time > time_threshold)  # how many orders will be over time
                    original_total_travel_time = np.sum(current_order_state[:, 3])
                    time_add = np.sum(new_total_travel_time) - original_total_travel_time  # total added time of all orders

                    reward_pick = reward_func(time_add,timeout,pick_pre_time2,direct_distance)
                    reward_pre += reward_pick * reward_parameter_list[6]

                    log["reward"] = reward_pick
                    log["wait"] = pick_pre_time2
                    log["pickup"] = pick_pre_time
                    log["workload"] = pick_pre_time + np.max(new_time) - np.max(current_order_state[:, 2])
                    log["direct"] = pick_pre_time + direct_time[0]
                    log["detour"] = new_time[-1] - direct_time[0] + np.sum(new_time[:-1]) - np.sum(current_order_state[:, 2]) + pick_pre_time * current_order_num

                    return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], None, current_time], [reward_pre, reward], pick_pre_time], new_route, new_route_time, new_time, new_total_travel_time, assign_state, log  # in this circumstance, the assignment must be None
                else:
                    reward_pre = 0

            else: # if the pre-booked order does not allow pooling
                if current_order_num == 0: # if no unfinished order
                    _, _, pick_pre_time, _ = TSP_route((curr_lat, curr_lon), [(observe[2], observe[3])])
                    pick_pre_time = pick_pre_time[0]
                    arrive_time = pick_pre_time + current_time
                    if arrive_time >= observe[6]: # start to pick up the pre-booked order
                        assign_state = 1
                        if arrive_time > observe[6]:
                            reward_pre = - reward_parameter_list[1] * (arrive_time - observe[6])
                            reward_pre *= reward_parameter_list[5]
                            # if current_time > observe[6]:
                            #     reward_pre *= reward_parameter_list[2]
                        elif arrive_time == observe[6]:
                            reward_pre = reward_parameter_list[0]

                        new_route, new_route_time, new_time, direct_distance = TSP_route((observe[2], observe[3]),[(observe[4], observe[5])])
                        pick_pre_time2 = arrive_time - observe[6]
                        # no previous order
                        new_total_travel_time = np.array(new_time) + pick_pre_time2
                        time_add = np.sum(new_total_travel_time)
                        timeout = int(new_total_travel_time > time_threshold)

                        reward_pick = reward_func(time_add, timeout, pick_pre_time2, direct_distance)
                        reward_pre += reward_pick * reward_parameter_list[6]

                        log["reward"] = reward_pick
                        log["wait"] = pick_pre_time2
                        log["pickup"] = pick_pre_time
                        log["workload"] = pick_pre_time + new_time[0]
                        log["direct"] = pick_pre_time +  new_time[0]
                        log["detour"] = 0

                        return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], None, current_time], [reward_pre, reward], pick_pre_time], new_route, new_route_time, new_time, new_total_travel_time, assign_state, log  # in this circumstance, the assignment must be None
                    else:
                        reward_pre = 0
                else:
                    # need to wait until all orders finish
                    last_order_index = np.argmax(current_order_state[:, 2])
                    rest_finishing_time = current_order_state[last_order_index, 2]
                    dest_lat, dest_lon = current_order_state[last_order_index, 0], current_order_state[last_order_index, 1]
                    _, _, pick_pre_time, _ = TSP_route((dest_lat, dest_lon), [(observe[2], observe[3])])
                    pick_pre_time = pick_pre_time[0]
                    arrive_time = pick_pre_time + rest_finishing_time + current_time
                    if arrive_time > observe[6]:  # add overtime punishment for pre-booked order
                        reward_pre = - reward_parameter_list[1] * (arrive_time - observe[6]) # currently, the worker may be still assigned an on-demand order
                        if current_time > observe[6]:
                            reward_pre *= reward_parameter_list[2]
                    else:
                        reward_pre = 0

    if worker_type == 0 and assignment is not None: # assign on-demand order (worker_type must be 0)
        plat, plon, dlat, dlon, appear_time, type = new_orders_state[assignment]
        waiting_time = current_time - appear_time

        pickup_route, pickup_route_t, pickup_time, _ = TSP_route((curr_lat, curr_lon), [(plat, plon)])
        pickup_time = pickup_time[0]
        direct_route, direct_route_time, direct_time, direct_distance = TSP_route((plat, plon), [(dlat, dlon)])
        direct_time = direct_time[0]

        # 1. detect the conflict
        if type == 1: # if the new on-demand order does not allow pooling
            if current_order_num != 0: # reject the on-demand order if current seat is not empty
                reward = 0
                return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], new_orders_state[assignment], current_time], [reward_pre, reward], None], None, None, None, None, assign_state, log

            if order_pre is not None:  # has pre-booked order
                _, _, pick_pre_time, _ = TSP_route((dlat, dlon), [(observe[2], observe[3])])
                pick_pre_time = pick_pre_time[0]
                arrive_time = pickup_time + direct_time + pick_pre_time + current_time
                if arrive_time > observe[6]: # reject the on-demand order if any conflict exists
                    # reward = 0
                    reward = - reward_parameter_list[3] * (arrive_time - observe[6])
                    return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], new_orders_state[assignment], current_time],[reward_pre,reward], None], None, None, None, None, assign_state, log

        else: # if the new on-demand order allows pooling
            if order_pre is not None:  # has pre-booked order
                order_pre_type = observe[7]
                if order_pre_type == 1:  # if the pre-booked order does not allow pooling
                    # wait until current order finishes
                    _, _, pick_pre_time, _ = TSP_route((dlat, dlon), [(observe[2], observe[3])])
                    pick_pre_time = pick_pre_time[0]
                    arrive_time = pickup_time + direct_time + pick_pre_time + current_time
                else:  # if the pre-booked order allows pooling
                    # wait until current order gets picked
                    _, _, pick_pre_time, _ = TSP_route((plat, plon), [(observe[2], observe[3])])
                    pick_pre_time = pick_pre_time[0]
                    arrive_time = pickup_time + pick_pre_time + current_time

                if arrive_time > observe[6]: # reject the on-demand order if any conflict exists
                    # reward = 0
                    reward = - reward_parameter_list[3] * (arrive_time - observe[6])
                    return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], new_orders_state[assignment], current_time],[reward_pre,reward], None], None, None, None, None, assign_state, log

        # Then we can assign the new order as traditional pooling scenario
        assign_state = 2
        destination_points = []
        for i in range(current_order_num):
            destination_points.append((current_order_state[i, 0], current_order_state[i, 1]))
        destination_points.append((dlat, dlon))
        new_route, new_route_time, new_time, _ = TSP_route((plat, plon), destination_points)
        new_total_travel_time = np.array(new_time)
        new_total_travel_time[:-1] = new_total_travel_time[:-1] + current_order_state[:current_order_num, 3] - current_order_state[:current_order_num, 2]  # add the time already cost for each old order
        new_total_travel_time[:-1] = new_total_travel_time[:-1] + pickup_time
        pickup_time2 = pickup_time + waiting_time
        new_total_travel_time[-1] += pickup_time2

        original_total_travel_time = np.sum(current_order_state[:, 3])
        time_add = np.sum(new_total_travel_time) - original_total_travel_time  # total added time of all orders
        timeout = np.sum(new_total_travel_time > time_threshold)  # how many orders will be over time
        reward = reward_func(time_add,timeout,pickup_time2,direct_distance)
        if reward_pre is not None:
            reward_pre += reward * reward_parameter_list[4]

        log["reward"] = reward
        log["confirmation"] = waiting_time
        log["wait"] = pickup_time2
        log["pickup"] = pickup_time
        log["workload"] = pickup_time + np.max(new_time) - np.max(current_order_state[:, 2])
        log["direct"] = pickup_time + direct_time
        log["detour"] = new_time[-1] - direct_time + np.sum(new_time[:-1]) - np.sum(current_order_state[:, 2]) + pickup_time * current_order_num

        return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], new_orders_state[assignment], current_time],[reward_pre,reward], pickup_time], new_route, new_route_time, new_time, new_total_travel_time, assign_state, log

    return [[[observe_pre, current_order_state, current_order_num], order_pre, [observe, current_order_state, current_order_num], None, current_time], [reward_pre, reward], None], None, None, None, None, assign_state, log
