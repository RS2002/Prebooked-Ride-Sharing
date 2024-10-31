from Worker import Buffer, Worker
from Platform import Platform, reward_func_generator
from Order_Env import Demand
import argparse
import tqdm
import torch
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_times', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--converge_epoch', type=int, default=10)
    parser.add_argument('--minimum_episode', type=int, default=500)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=10000)
    parser.add_argument('--demand_sample_rate', type=float, default=0.95)
    parser.add_argument('--prebooked_sample_rate', type=float, default=0.20)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[5.0,3.0,2.0,4.0])
    parser.add_argument('--punishment', type=float, default=10.0)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--arl", action="store_true",default=True)
    parser.add_argument('--eval_episode', type=int, default=10)

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=48)

    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--model_pre_path",type=str,default=None)

    parser.add_argument("--demand_path",type=str,default="./data/yellow_tripdata_2024-07.parquet")
    parser.add_argument("--zone_dic_path",type=str,default="./data/Manhattan_dic.pkl")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    exploration_rate = args.epsilon
    epsilon_decay_rate = args.epsilon_decay_rate
    epsilon_final = args.epsilon_final

    with open(args.zone_dic_path, 'rb') as f:
        zone_dic = pickle.load(f)
    zone_table = zone_dic["zone_num"]

    platform = Platform(discount_factor=args.gamma, njobs=args.njobs)
    demand = Demand(demand_path=args.demand_path, zone_table = zone_table)
    buffer = Buffer(capacity=args.buffer_capacity)
    buffer_pre = Buffer(capacity=args.buffer_capacity)
    worker = Worker(buffer, buffer_pre, lr=args.lr, gamma=args.gamma, max_step=args.max_step, num=args.worker_num, device=device, zone_table_path = args.zone_dic_path, model_path = args.model_path, model_pre_path = args.model_pre_path, njobs = args.njobs, arl = args.arl, dropout = args.dropout)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    best_reward = -1e-8
    best_epoch = 0

    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)

    while True:
        j += 1
        worker.reset(train=True)
        platform.reset(discount_factor=args.gamma)
        demand.reset(day = 1, hour = 7, start_time = 0,  pre_sample = args.prebooked_sample_rate, p_sample = args.demand_sample_rate, wait_time = args.order_max_wait_time)

        exploration_rate = max(exploration_rate * epsilon_decay_rate, epsilon_final)
        print("Exploration Rate: ", exploration_rate)

        picked_ondemand_orders = 0
        picked_prebook_orders = 0

        pbar = tqdm.tqdm(range(args.max_step))
        for t in pbar:
            q_value, order_pre = worker.observe(worker.Q_training_pre, demand.current_demand_pre, t, exploration_rate)
            assignment_pre, _ = platform.assign(q_value)
            observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
            q_value, order = worker.observe(worker.Q_training, demand.current_demand, t, exploration_rate)
            assignment, _ = platform.assign(q_value)
            feedback_table, new_route_table, new_route_time_table, accepted_pre, accepted_on, assign_state_table = platform.feedback(observe_pre, order_pre, assignment_pre, worker.observe_space, assignment, order, reward_func, args.punishment, t)
            worker.update(feedback_table, new_route_table, new_route_time_table, assign_state_table, (t == args.max_step - 1), j)
            demand.pickup(accepted_on, accepted_pre)
            demand.update()
            picked_ondemand_orders += len(accepted_on)
            picked_prebook_orders += len(accepted_pre)
        loss = worker.train(buffer,worker.Q_training,worker.Q_target,worker.optim,worker.schedule,batch_size=args.batch_size,train_times=args.train_times)
        loss_pre = worker.train(buffer_pre,worker.Q_training_pre,worker.Q_target_pre,worker.optim_pre,worker.schedule_pre,batch_size=args.batch_size,train_times=args.train_times)
        worker.update_Qtarget()

        overtime_num = platform.overtime_num
        average_overtime = platform.overtime / (picked_prebook_orders+1e-8)
        total_reward = platform.Total_Reward
        total_reward_pre = platform.Total_Reward_Pre

        log = "Train Episode {:} , On-demand Reward {:} , Pre-booked Reward {:} , On-demand Pickup {:} , Pre-booked Pickup {:} , Pre-booked Overtime_num {:} , Average Overtime {:} , On-demand Loss {:} , Pre-booked Loss {:}".format(
            j, total_reward, total_reward_pre, picked_ondemand_orders,picked_prebook_orders,overtime_num,average_overtime,loss,loss_pre)
        print(log)
        with open("train.txt", 'a') as file:
            file.write(log + "\n")
        worker.save("latest.pth", "latest_pre.pth")


        if j % args.eval_episode == 0:
            worker.reset(train=False)
            platform.reset(discount_factor=args.gamma)
            demand.reset(day=1, hour=7, start_time=0, pre_sample=args.prebooked_sample_rate,
                         p_sample=args.demand_sample_rate, wait_time=args.order_max_wait_time)

            picked_ondemand_orders = 0
            picked_prebook_orders = 0

            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, order_pre = worker.observe(worker.Q_training_pre, demand.current_demand_pre, t,
                                                    0)
                assignment_pre, _ = platform.assign(q_value)
                observe_pre, order_pre = worker.update_pre(assignment_pre, order_pre)
                q_value, order = worker.observe(worker.Q_training, demand.current_demand, t, 0)
                assignment, _ = platform.assign(q_value)
                feedback_table, new_route_table, new_route_time_table, accepted_pre, accepted_on, assign_state_table = platform.feedback(
                    observe_pre, order_pre, assignment_pre, worker.observe_space, assignment, order, reward_func,
                    args.punishment, t)
                worker.update(feedback_table, new_route_table, new_route_time_table, assign_state_table,
                              (t == args.max_step - 1), j)
                demand.pickup(accepted_on, accepted_pre)
                demand.update()
                picked_ondemand_orders += len(accepted_on)
                picked_prebook_orders += len(accepted_pre)
            overtime_num = platform.overtime_num
            average_overtime = platform.overtime / (picked_prebook_orders+1e-8)
            total_reward = platform.Total_Reward
            total_reward_pre = platform.Total_Reward_Pre

            log = "Eval Episode {:} , On-demand Reward {:} , Pre-booked Reward {:} , On-demand Pickup {:} , Pre-booked Pickup {:} , Pre-booked Overtime_num {:} , Average Overtime {:}".format(
                j, total_reward, total_reward_pre, picked_ondemand_orders,picked_prebook_orders,overtime_num,average_overtime)
            print(log)
            with open("eval.txt", 'a') as file:
                file.write(log + "\n")

        total_reward += total_reward_pre
        if total_reward > best_reward:
            best_epoch = 0
            best_reward = total_reward
            worker.save("best.pth", "best_pre.pth")
        else:
            best_epoch += 1
        if j == args.minimum_episode:
            best_epoch = 0
        elif j > args.minimum_episode:
            print("Converge Step: ", best_epoch)
            if best_epoch >= args.converge_epoch:
                break

if __name__ == '__main__':
    main()