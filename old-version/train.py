from Worker import Buffer, Worker
from Platform import Platform, reward_func_generator, assign
from Order_Env import Demand
import argparse
import tqdm
import torch
import pickle
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--max_episode', type=int, default=1000)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=1e8)
    parser.add_argument('--buffer_episode', type=int, default=20)
    parser.add_argument('--demand_sample_rate', type=float, default=0.95)
    parser.add_argument('--prebooked_rate', type=float, default=0.80)
    parser.add_argument('--pooling_rate', type=float, default=0.80)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[5.0,3.0,4.0,2.0,1.0,3.0])
    parser.add_argument('--reward_parameter2', type=float, nargs='+', default=[10.0,3.0,1.0,1.0,0.0,1.0,1.0,1.0,3.0])
    parser.add_argument("--mode", type=int, default=2) # 2 -> all pre-booked orders appear at (schedule_time-advance_time)
    parser.add_argument("--prebook_start", type=int, default=0)
    parser.add_argument("--prebook_end", type=int, default=0)
    parser.add_argument("--advance_time", type=int, default=30)
    parser.add_argument("--rand_mode", action="store_true",default=False)
    parser.add_argument("--rand_rate", action="store_true",default=False) # random pre-booked & pooling rate
    parser.add_argument("--rand_appear", action="store_true",default=False) # random pre-booked orders appearing time
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--bi_direction", action="store_true",default=False)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=12)
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--demand_path",type=str,default="../data/yellow_tripdata_2024-07.parquet")
    parser.add_argument("--zone_dic_path",type=str,default="../data/Manhattan_dic.pkl")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    # day = args.day
    hour = args.hour
    train_list = []
    eval_list = []
    exploration_rate = args.epsilon
    epsilon_decay_rate = args.epsilon_decay_rate
    epsilon_final = args.epsilon_final

    with open(args.zone_dic_path, 'rb') as f:
        zone_dic = pickle.load(f)
    zone_table = zone_dic["zone_num"]

    platform = Platform(discount_factor=args.gamma, njobs=args.njobs)
    demand = Demand(demand_path=args.demand_path, zone_table = zone_table)
    buffer = Buffer(capacity=args.buffer_capacity, episode_capacity = args.buffer_episode)
    worker = Worker(buffer, lr=args.lr, gamma=args.gamma, max_step=args.max_step, num=args.worker_num, device=device, zone_table_path = args.zone_dic_path, model_path = args.model_path, njobs = args.njobs, bi_direction = args.bi_direction, dropout = args.dropout)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)
    train_pre = True

    prebooked_rate = args.prebooked_rate
    pooling_rate = args.pooling_rate
    rand_prop = args.rand_rate
    rand_appear = args.rand_appear
    rand_mode = args.rand_mode

    while j<=args.max_episode:
        j += 1
        # set pre-booked rate, pooling rate, and mode
        if rand_prop:
            pre_sample = random.random()
            p_pooling = random.random()
            print("Pre-booked Rate: {:} , Pooling Rate: {:}".format(pre_sample,p_pooling))
        else:
            pre_sample = prebooked_rate
            p_pooling = pooling_rate
        if rand_mode:
            mode = random.randint(1, 3)
            print("Mode {:}".format(mode))
        else:
            mode = args.mode
        if rand_appear:
            prebook_start = random.randint(0, 10)
            prebook_end = random.randint(prebook_start,20)
            # advance_time = random.randint(10, 60)
            advance_time = random.randint(10, 30)
            print("Pre-booked Start Time: {:} , End Time: {:} , Advance Time {:}".format(prebook_start,prebook_end,advance_time))
        else:
            prebook_start = args.prebook_start
            prebook_end = args.prebook_end
            advance_time = args.advance_time

        worker.reset(train=True, pre_rate=pre_sample, pooling_rate=p_pooling)
        platform.reset(discount_factor=args.gamma)
        # day = random.randint(8,12)
        day = args.day
        ondemand_pooling, ondemand_nonpooling, prebooked_pooling, prebooked_nonpooling = demand.reset(day = day, hour = hour, start_time = 0,  pre_sample = pre_sample, p_sample = args.demand_sample_rate, p_pooling = p_pooling, prebook_start = prebook_start, prebook_end = prebook_end, wait_time = args.order_max_wait_time, mode = mode, advance_time = advance_time)

        exploration_rate = max(exploration_rate * epsilon_decay_rate, epsilon_final)

        print("Exploration Rate: ", exploration_rate)
        pbar = tqdm.tqdm(range(args.max_step))
        loss_list = []

        for t in pbar:
            q_value, order_pre = worker.observe(True, demand.current_demand_pre, t, None, exploration_rate)
            if order_pre is not None:
                assignment_pre, _ = assign(q_value,pad=False)
            else:
                assignment_pre = [None] * worker.num
            observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
            q_value, order = worker.observe(False, demand.current_demand, t, None, exploration_rate)
            if order is not None:
                assignment, _ = assign(q_value)
            else:
                assignment = [None] * worker.num
            feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, assign_state_table, accepted_pre, accepted_on = platform.feedback(observe_pre, order_pre, assignment_pre, worker.observe_space, worker.current_orders, worker.current_order_num, assignment, order, args.order_max_wait_time, reward_func, args.reward_parameter2, t)
            worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, assign_state_table, (t == args.max_step - 1), j)
            demand.pickup(accepted_on, accepted_pre)
            demand.update()

            if (t+1) % 4 == 0:
                if buffer.num > args.batch_size:
                    loss = worker.train(batch_size=args.batch_size, train_times=1, show_pbar=False)
                    loss_list.append(loss)

        loss = np.mean(loss_list)
        worker.schedule.step()
        train_pre = not train_pre

        total_demand = np.array([prebooked_pooling, prebooked_nonpooling, ondemand_pooling, ondemand_nonpooling])
        drop_demand = np.array([demand.num_lost_demand_pooling, demand.num_lost_demand_nonpooling])
        drop_demand_rate = drop_demand / total_demand[2:]
        max_utilization_rate = worker.max_utilization_rate
        total_utilization_rate = np.mean(worker.start_flag)
        Pickup_Num = np.array(platform.Pickup_Num)
        Detour = np.array(platform.Detour)
        Overtime_num = np.array(platform.Overtime_num)
        Overtime = np.array(platform.Overtime)
        service_rate = Pickup_Num / total_demand
        average_detour = Detour / Pickup_Num
        overtime_average = Overtime / Pickup_Num[:2]
        overtime_rate = (Overtime_num + (total_demand[:2] - Pickup_Num[:2])) / total_demand[:2]
        reward = platform.Total_Reward / args.worker_num
        reward_pre = platform.Total_Reward_Pre / args.worker_num
        idle_time = worker.waiting_time_list
        idle_time = np.mean(idle_time)
        pickup_time = np.array(platform.Pickup_Time)
        pickup_time = pickup_time / Pickup_Num
        type_reward = np.array(platform.Reward)
        total_reward = np.sum(type_reward) / args.worker_num
        unit_reward = type_reward / Pickup_Num
        swap_rate = platform.Swap / (platform.Assigment + 1e-8)

        print("Train Episode {:}, On-demand Reward {:}, Pre-booked Reward {:}, Total Reward {:}, Idle Time {:}, Utilization Rate {:}, Swap Rate {:}, Loss {:}".format(j, reward, reward_pre,total_reward,idle_time, max_utilization_rate, swap_rate, loss))
        print("Service Rate: ", service_rate)
        print("Loss Demand: ", drop_demand_rate)
        print("Average Detour: ", average_detour)
        print("Overtime Rate: ", overtime_rate)
        print("Average Overtime: ", overtime_average)
        print("Pickup Time: ", pickup_time)
        print("Unit Reward: ", unit_reward)
        print()
        worker.save("model.pth")
        dic = {
            'episode': j,
            'service_rate': service_rate,
            'drop_demand_rate': drop_demand_rate,
            'max_utilization_rate': max_utilization_rate,
            'total_utilization_rate': total_utilization_rate,
            'average_detour': average_detour,
            'overtime_rate': overtime_rate,
            'overtime_average': overtime_average,
            "pickup_time": pickup_time,
            'reward': reward,
            'reward_pre': reward_pre,
            "total_reward": total_reward,
            "unit_reward": unit_reward,
            "swap_rate": swap_rate,
            "idle_time": idle_time,
            "total_demand": total_demand,
            "pickup_num": Pickup_Num,
            'loss': loss,
        }
        train_list.append(dic)
        with open('train.pkl', 'wb') as f:
            pickle.dump(train_list, f)

        if j % args.eval_episode == 0:
            pre_sample = prebooked_rate
            p_pooling = pooling_rate
            mode = args.mode
            prebook_start = args.prebook_start
            prebook_end = args.prebook_end
            advance_time = args.advance_time
            print("Pre-booked Rate: {:} , Pooling Rate: {:}".format(pre_sample,p_pooling))
            print("Mode {:}".format(mode))
            print("Pre-booked Start Time: {:} , End Time: {:} , Advance Time {:}".format(prebook_start,prebook_end,advance_time))
            worker.reset(train=False, pre_rate=pre_sample, pooling_rate=p_pooling)
            platform.reset(discount_factor=args.gamma)
            day = args.day
            ondemand_pooling, ondemand_nonpooling, prebooked_pooling, prebooked_nonpooling = demand.reset(day=day, hour=hour,
                                                                                                          start_time=0,
                                                                                                          pre_sample=pre_sample,
                                                                                                          p_sample=args.demand_sample_rate,
                                                                                                          p_pooling=p_pooling,
                                                                                                          prebook_start = prebook_start,
                                                                                                          prebook_end = prebook_end,
                                                                                                          wait_time=args.order_max_wait_time,
                                                                                                          mode = mode,
                                                                                                          advance_time = advance_time)
            print("Exploration Rate: ", 0)
            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, order_pre = worker.observe(True, demand.current_demand_pre, t,
                                                    None, 0)
                if order_pre is not None:
                    assignment_pre, _ = assign(q_value,pad=False)
                else:
                    assignment_pre = [None] * worker.num
                observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
                q_value, order = worker.observe(False, demand.current_demand, t, None, 0)
                if order is not None:
                    assignment, _ = assign(q_value)
                else:
                    assignment = [None] * worker.num
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, assign_state_table, accepted_pre, accepted_on = platform.feedback(
                    observe_pre, order_pre, assignment_pre, worker.observe_space, worker.current_orders,
                    worker.current_order_num, assignment, order, args.order_max_wait_time, reward_func,
                    args.reward_parameter2, t)
                worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                              new_total_travel_time_table, assign_state_table, (t == args.max_step - 1), j)
                demand.pickup(accepted_on, accepted_pre)
                demand.update()

            total_demand = np.array(
                [prebooked_pooling, prebooked_nonpooling, ondemand_pooling, ondemand_nonpooling])  # + 1e-8
            drop_demand = np.array([demand.num_lost_demand_pooling, demand.num_lost_demand_nonpooling])
            drop_demand_rate = drop_demand / total_demand[2:]
            max_utilization_rate = worker.max_utilization_rate
            total_utilization_rate = np.mean(worker.start_flag)
            Pickup_Num = np.array(platform.Pickup_Num)
            Detour = np.array(platform.Detour)
            Overtime_num = np.array(platform.Overtime_num)
            Overtime = np.array(platform.Overtime)
            service_rate = Pickup_Num / total_demand
            average_detour = Detour / Pickup_Num
            overtime_average = Overtime / Pickup_Num[:2]
            overtime_rate = (Overtime_num + (total_demand[:2] - Pickup_Num[:2])) / total_demand[:2]
            reward = platform.Total_Reward / args.worker_num
            reward_pre = platform.Total_Reward_Pre / args.worker_num
            idle_time = worker.waiting_time_list
            idle_time = np.mean(idle_time)
            pickup_time = np.array(platform.Pickup_Time)
            pickup_time = pickup_time / Pickup_Num
            type_reward = np.array(platform.Reward)
            total_reward = np.sum(type_reward) / args.worker_num
            unit_reward = type_reward / Pickup_Num
            swap_rate = platform.Swap / (platform.Assigment + 1e-8)
            print(
                "Eval Episode {:}, On-demand Reward {:}, Pre-booked Reward {:}, Total Reward {:}, Idle Time {:}, Utilizationb Rate {:}, Swap Rate {:}".format(
                    j, reward, reward_pre, total_reward, idle_time, max_utilization_rate, swap_rate))
            print("Service Rate: ", service_rate)
            print("Loss Demand: ", drop_demand_rate)
            print("Average Detour: ", average_detour)
            print("Overtime Rate: ", overtime_rate)
            print("Average Overtime: ", overtime_average)
            print("Pickup Time: ", pickup_time)
            print("Unit Reward: ", unit_reward)
            print()
            dic = {
                'episode': j,
                'service_rate': service_rate,
                'drop_demand_rate': drop_demand_rate,
                'max_utilization_rate': max_utilization_rate,
                'total_utilization_rate': total_utilization_rate,
                'average_detour': average_detour,
                'overtime_rate': overtime_rate,
                'overtime_average': overtime_average,
                "pickup_time": pickup_time,
                'reward': reward,
                'reward_pre': reward_pre,
                "total_reward": total_reward,
                "unit_reward": unit_reward,
                "swap_rate": swap_rate,
                "idle_time": idle_time,
                "total_demand": total_demand,
                "pickup_num": Pickup_Num
            }
            eval_list.append(dic)
            with open('eval.pkl', 'wb') as f:
                pickle.dump(eval_list, f)

            print()


if __name__ == '__main__':
    main()