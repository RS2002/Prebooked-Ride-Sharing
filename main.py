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

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_times', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--converge_epoch', type=int, default=10)
    parser.add_argument('--minimum_episode', type=int, default=1000)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=5e5)
    parser.add_argument('--buffer_episode', type=int, default=10)
    parser.add_argument('--demand_sample_rate', type=float, default=0.95)
    parser.add_argument('--prebooked_rate', type=float, default=0.20)
    parser.add_argument('--pooling_rate', type=float, default=0.80)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[5.0,3.0,4.0,2.0,1.0,3.0])
    parser.add_argument('--reward_parameter2', type=float, nargs='+', default=[15.0,1.0,2.0,1.0,0.0,5.0,3.0])

    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--prebook_start", type=int, default=0)
    parser.add_argument("--prebook_end", type=int, default=10)
    parser.add_argument("--advance_time", type=int, default=20)
    parser.add_argument("--rand_mode", action="store_true",default=False)
    parser.add_argument("--rand_rate", action="store_true",default=False)
    parser.add_argument("--rand_appear", action="store_true",default=False)


    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--bi_direction", action="store_true",default=False)
    parser.add_argument('--eval_episode', type=int, default=10)

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=24)

    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--model_pre_path",type=str,default=None)

    parser.add_argument("--demand_path",type=str,default="../data/yellow_tripdata_2024-07.parquet")
    parser.add_argument("--zone_dic_path",type=str,default="../data/Manhattan_dic.pkl")

    parser.add_argument("--follow", action="store_true",default=False)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

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
    buffer_pre = Buffer(capacity=args.buffer_capacity, episode_capacity = args.buffer_episode)
    worker = Worker(buffer, buffer_pre, lr=args.lr, gamma=args.gamma, max_step=args.max_step, num=args.worker_num, device=device, zone_table_path = args.zone_dic_path, model_path = args.model_path, model_pre_path = args.model_pre_path, njobs = args.njobs, bi_direction = args.bi_direction, dropout = args.dropout)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    best_reward = -100.0
    best_epoch = 0
    best_reward_on = -100.0
    best_reward_pre = -100.0

    flag_on = False
    flag_pre = False


    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)

    exploration_rate_on = exploration_rate
    exploration_rate_pre = exploration_rate

    prebooked_rate = args.prebooked_rate
    pooling_rate = args.pooling_rate

    train_pre = True
    rand_prop = args.rand_rate
    rand_appear = args.rand_appear
    rand_mode = args.rand_mode

    while True:

        if flag_on:
            train_pre = False
        elif flag_pre:
            train_pre = True
        else:
            train_pre = not train_pre

        j += 1
        worker.reset(train=True, train_pre=train_pre)
        platform.reset(discount_factor=args.gamma)

        if rand_prop:
            # pre_sample = random.random()
            # p_pooling = random.random()

            p_pooling = random.randint(0,10)
            p_pooling = p_pooling * 0.1
            if train_pre:
                pre_sample = random.randint(1,10)
            else:
                pre_sample = random.randint(0,10)
            pre_sample = pre_sample * 0.1

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

            # advance_time = random.randint(10,30)
            advance_time = random.randint(1, 3)
            advance_time = advance_time * 10

            print("Pre-booked Start Time: {:} , End Time: {:} , Advance Time {:}".format(prebook_start,prebook_end,advance_time))
        else:
            prebook_start = args.prebook_start
            prebook_end = args.prebook_end
            advance_time = args.advance_time



        ondemand_pooling, ondemand_nonpooling, prebooked_pooling, prebooked_nonpooling = demand.reset(day = 1, hour = 7, start_time = 0,  pre_sample = pre_sample, p_sample = args.demand_sample_rate, p_pooling = p_pooling, prebook_start = prebook_start, prebook_end = prebook_end, wait_time = args.order_max_wait_time, mode = mode, advance_time = advance_time)


        if train_pre:
            exploration_rate_pre = max(exploration_rate_pre * epsilon_decay_rate, epsilon_final)
            exploration_rate_temp1 = exploration_rate_pre
            exploration_rate_temp2 = 0
        else:
            exploration_rate_temp1 = 0
            exploration_rate_on = max(exploration_rate_on * epsilon_decay_rate, epsilon_final)
            exploration_rate_temp2 = exploration_rate_on


        print("Exploration Rate: ", exploration_rate_temp1, exploration_rate_temp2)
        pbar = tqdm.tqdm(range(args.max_step))
        for t in pbar:
            q_value, order_pre = worker.observe(worker.Q_training_pre, demand.current_demand_pre, t, None, exploration_rate_temp1)
            if order_pre is not None:
                assignment_pre, _ = assign(q_value)
            else:
                assignment_pre = [None] * worker.num
            observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
            q_value, order = worker.observe(worker.Q_training, demand.current_demand, t, None, exploration_rate_temp2)
            if order is not None:
                assignment, _ = assign(q_value)
            else:
                assignment = [None] * worker.num
            feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, assign_state_table, accepted_pre, accepted_on = platform.feedback(observe_pre, order_pre, assignment_pre, worker.observe_space, worker.current_orders, worker.current_order_num, assignment, order, args.order_max_wait_time, reward_func, args.reward_parameter2, t)
            worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, assign_state_table, (t == args.max_step - 1), j)
            demand.pickup(accepted_on, accepted_pre)
            demand.update()

        if train_pre:
            loss = 0
            loss_pre = worker.train(buffer_pre,worker.Q_training_pre,worker.Q_target_pre,worker.optim_pre,worker.schedule_pre,batch_size=args.batch_size,train_times=args.train_times)
            worker.update_Q_pre()
        else:
            loss = worker.train(buffer,worker.Q_training,worker.Q_target,worker.optim,worker.schedule,batch_size=args.batch_size,train_times=args.train_times)
            loss_pre = 0
            worker.update_Q_on()

        total_demand = np.array([prebooked_pooling, prebooked_nonpooling, ondemand_pooling, ondemand_nonpooling]) # + 1e-8
        drop_demand = np.array([demand.num_lost_demand_pooling, demand.num_lost_demand_nonpooling])
        drop_demand_rate = drop_demand / total_demand[2:]
        max_ultilization_rate = worker.max_ultilization_rate
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

        # idle_time = worker.idle_time
        idle_time = worker.waiting_time_list
        idle_time = np.mean(idle_time)
        pickup_time = np.array(platform.Pickup_Time)
        pickup_time = pickup_time / Pickup_Num
        type_reward = np.array(platform.Reward)
        total_reward = np.sum(type_reward) / args.worker_num
        unit_reward = type_reward / Pickup_Num
        swap_rate = platform.Swap / (platform.Assigment + 1e-8)


        print("Train Episode {:}, On-demand Reward {:}, Pre-booked Reward {:}, Total Reward {:}, Idle Time {:}, Ultilizationb Rate {:}, Swap Rate {:},  On-demand Loss {:}, Pre-booked Loss {:}".format(j, reward, reward_pre,total_reward,idle_time, max_ultilization_rate, swap_rate, loss, loss_pre))
        print("Service Rate: ", service_rate)
        print("Loss Demand: ", drop_demand_rate)
        print("Average Detour: ", average_detour)
        print("Overtime Rate: ", overtime_rate)
        print("Average Overtime: ", overtime_average)
        print("Pickup Time: ", pickup_time)
        print("Unit Reward: ", unit_reward)
        print()
        worker.save("latest.pth", "latest_pre.pth")
        dic = {
            'episode': j,
            'service_rate': service_rate,
            'drop_demand_rate': drop_demand_rate,
            'max_ultilization_rate': max_ultilization_rate,
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
            'loss_pre': loss_pre
        }
        train_list.append(dic)
        with open('train.pkl', 'wb') as f:
            pickle.dump(train_list, f)

        # if j % (2*args.eval_episode) == 0:
        #     worker.update_Qtarget(1.0)

        if j % args.eval_episode == 0:

            # train_pre = not train_pre
            # buffer.reset()
            # buffer_pre.reset()

            worker.reset(train=False)
            platform.reset(discount_factor=args.gamma)

            pre_sample = prebooked_rate
            p_pooling = pooling_rate
            mode = args.mode
            # mode = 2
            prebook_start = args.prebook_start
            prebook_end = args.prebook_end
            advance_time = args.advance_time

            ondemand_pooling, ondemand_nonpooling, prebooked_pooling, prebooked_nonpooling = demand.reset(day=1, hour=7,
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
                q_value, order_pre = worker.observe(worker.Q_training_pre, demand.current_demand_pre, t,
                                                    None, 0)
                if order_pre is not None:
                    assignment_pre, _ = assign(q_value)
                else:
                    assignment_pre = [None] * worker.num
                observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
                q_value, order = worker.observe(worker.Q_training, demand.current_demand, t, None, 0)
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
            max_ultilization_rate = worker.max_ultilization_rate
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

            # idle_time = worker.idle_time
            idle_time = worker.waiting_time_list
            idle_time = np.mean(idle_time)
            pickup_time = np.array(platform.Pickup_Time)
            pickup_time = pickup_time / Pickup_Num
            type_reward = np.array(platform.Reward)
            total_reward = np.sum(type_reward) / args.worker_num
            unit_reward = type_reward / Pickup_Num
            swap_rate = platform.Swap / (platform.Assigment + 1e-8)
            print(
                "Eval Episode {:}, On-demand Reward {:}, Pre-booked Reward {:}, Total Reward {:}, Idle Time {:}, Ultilizationb Rate {:}, Swap Rate {:}".format(
                    j, reward, reward_pre, total_reward, idle_time, max_ultilization_rate, swap_rate))
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
                'max_ultilization_rate': max_ultilization_rate,
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


            if args.follow:
                reward_threshold_on = 2.0
                reward_threshold_pre = 1.0
                flag_pre = False
                flag_on = False

                if best_reward_pre - reward_pre > reward_threshold_pre:
                    print(best_reward_pre,reward_pre)
                    flag_pre = True
                    # worker.load("latest.pth", "latest_pre.pth",device)
                    buffer.reset()
                    buffer_pre.reset()
                    print("Train Pre-book Only!")
                elif best_reward_on - reward > reward_threshold_on:
                    print(best_reward_on,reward)
                    flag_on = True
                    # worker.load("latest.pth", "latest_pre.pth",device)
                    buffer.reset()
                    buffer_pre.reset()
                    print("Train On-demand Only!")
                else:
                    # worker.save("latest.pth", "latest_pre.pth")
                    if reward_pre < best_reward_pre and reward < best_reward_on:
                      best_epoch += 1
                      if j > args.minimum_episode and best_epoch >= args.converge_epoch:
                            break
                    else:
                        worker.save("best.pth", "best_pre.pth")
                        best_epoch = 0
                        if reward_pre >= best_reward_pre:
                            best_reward_pre = reward_pre
                        if reward >= best_reward_on:
                            best_reward_on = reward
            else:
                if reward_pre < best_reward_pre and reward < best_reward_on:
                    best_epoch += 1
                    if j > args.minimum_episode and best_epoch >= args.converge_epoch:
                        break
                else:
                    worker.save("best.pth", "best_pre.pth")
                    best_epoch = 0
                    if reward_pre >= best_reward_pre:
                        best_reward_pre = reward_pre
                    if reward >= best_reward_on:
                        best_reward_on = reward
            if j == args.minimum_episode:
                best_epoch = 0


            # total_reward = reward + reward_pre * 2
            # if total_reward > best_reward:
            #     best_epoch = 0
            #     best_reward = total_reward
            #     worker.save("best.pth", "best_pre.pth")
            # else:
            #     best_epoch += 1
            # if j == args.minimum_episode:
            #     best_epoch = 0
            # elif j > args.minimum_episode:
            #     print("Converge Step: ", best_epoch)
            #     if best_epoch >= args.converge_epoch:
            #         break


            print()


if __name__ == '__main__':
    main()