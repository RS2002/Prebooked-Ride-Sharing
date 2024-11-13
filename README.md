# D³QN

**Article:** "Ride-hailing Vehicle Dispatching with a Mixture of On-demand and Pre-booked Requests: A Deep Reinforcement Learning Approach"



The detail of the network structure can be found in [Double-PDF](https://github.com/RS2002/Double-PDF).



## 1. Workflow

![](./img/main.png)



## 2. Preparation

### 2.1 Dataset





### 2.2 OSRM

Before running the code, you should  first start the docker of [OSRM](https://github.com/Project-OSRM/osrm-backend) (northeast region of US). To avoid conflicting with our other work, we change the port as 6000:

```shell

```





## 3. How to Run

### 3.0 Our Setting





### 3.1 Train

#### 3.1.1 Train for a General Strategy

```shell
python main.py --rand_rate --rand_appear
```





#### 3.1.2 Train for a Specific Strategy

```shell
python main.py --prebook_start <when pre-booked orders start to appear> --prebook_end <when all pre-booked orders have appeared> --pooling_rate <the proportioin of customers choose pooling> --prebooked_rate <the proportion of pre-booked order in the total orders>
```





### 3.2 Eval

```shell
python eval.py --prebooked_rate <list1> --pooling_rate <list2> --prebook_start <list3_1> --prebook_end <list3_2>
```





## 4. Citation

```

```

