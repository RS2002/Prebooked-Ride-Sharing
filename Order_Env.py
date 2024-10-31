import pandas as pd
import pickle

class Demand():
    def __init__(self, demand_path, zone_table):
        self.demand = pd.read_parquet(demand_path)
        self.demand['tpep_pickup_datetime'] = pd.to_datetime(self.demand['tpep_pickup_datetime'])
        self.demand['year'] = self.demand['tpep_pickup_datetime'].dt.year
        self.demand['month'] = self.demand['tpep_pickup_datetime'].dt.month
        self.demand['day'] = self.demand['tpep_pickup_datetime'].dt.day
        self.demand['hour'] = self.demand['tpep_pickup_datetime'].dt.hour
        self.demand['minute'] = self.demand['tpep_pickup_datetime'].dt.minute
        self.demand = self.demand[(self.demand["PULocationID"].isin(zone_table)) & (self.demand["DOLocationID"].isin(zone_table))]


    '''
    episode_time: start minute in this episode
    p_sample: randomly select 100p% samples from the dataset
    wait_time: the maximum waiting time of each order
    '''
    def reset(self, day = 1, hour = 18, start_time = 0, pre_sample = 0.1, p_sample = 0.95, wait_time = 5):
        self.filtered_demand = self.demand[(self.demand["day"]==day) & (self.demand["hour"]==hour)]
        self.filtered_demand = self.filtered_demand.sample(frac=p_sample).sort_index()

        demand_pre = self.filtered_demand[self.filtered_demand['minute'] > 30]
        demand_pre = demand_pre.copy()
        self.filtered_demand_pre = demand_pre.sample(frac=pre_sample)
        self.filtered_demand = self.filtered_demand.drop(self.filtered_demand_pre.index)
        self.filtered_demand, self.filtered_demand_pre = self.filtered_demand.sort_index(), self.filtered_demand_pre.sort_index()

        print("total number of on-demand order at this episode is:", len(self.filtered_demand))
        print("total number of pre-booked order at this episode is:", len(self.filtered_demand_pre))

        self.current_demand = self.filtered_demand.loc[self.filtered_demand['minute'] == start_time].reset_index(drop=True)
        self.current_demand_pre = self.filtered_demand_pre.copy().reset_index(drop=True)
        self.current_time = start_time
        self.num_lost_demand = 0
        self.wait_time = wait_time


    '''
    update the order in the next minute
    throw away orders waiting longer than <wait_time> minutes
    '''
    def update(self):
        self.current_time += 1
        self.current_demand = pd.concat(
            [self.current_demand, self.filtered_demand.loc[self.filtered_demand['minute'] == self.current_time]])
        self.current_demand = self.current_demand.reset_index(drop=True)
        # drop those orders that are not taken over <wait_time> minutes
        if self.current_time >= self.wait_time:
            self.num_lost_demand += len(self.current_demand[self.current_demand['minute'] <= (self.current_time - self.wait_time)])
            self.current_demand = self.current_demand.drop(
                index=self.current_demand[self.current_demand['minute'] <= (self.current_time - self.wait_time)].index).reset_index(
                drop=True)


    '''
    delete the accepted orders from current_demand list 
    '''
    def pickup(self,unique_r_ids, pre_ids):
        # Convert the set to a list
        unique_r_ids_list = list(unique_r_ids)
        # Drop rows whose index is in unique_r_ids_list
        self.current_demand = self.current_demand.drop(unique_r_ids_list)
        # Reset index
        self.current_demand = self.current_demand.reset_index(drop=True)

        pre_ids = list(pre_ids)
        self.current_demand_pre = self.current_demand_pre.drop(pre_ids)
        self.current_demand_pre = self.current_demand_pre.reset_index(drop=True)

if __name__ == '__main__':
    # test
    with open("./data/Manhattan_dic.pkl", 'rb') as f:
        manhattan_dic = pickle.load(f)
    zone_table = manhattan_dic["zone_num"]
    demand = Demand("./data/yellow_tripdata_2024-07.parquet",zone_table)
    demand.reset(day = 1, hour = 18, start_time = 0, pre_sample = 0.1, p_sample = 0.95, wait_time = 5)
    demand.pickup([],[])