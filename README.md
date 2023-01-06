# School-Bus-Route-Optimization

This repository contains code for constrained school bus route optimization problem in which we are given a set of customer nodes spanned across a region and a set of all possible bus stops in the region. Along with this we have one depot which is the start and the ending point of the buses. Also we have taken a maximum limit each customer can walk to reach the busstop from where the person will be picked. The constraints that are included while forming the route are:
1. Maximum walking distance from home to bus-stop for passengers
2. Maximum number of buses that can be used for picking up passengers
3. Maximum capacity of bus
4. Maximum distance limit that each bus can traverse
5. Minimum percentage occupancy of that each bus should have
6. Time window for each bus within with it should start and end its journey

<img src = "./teaser.PNG">

Our solution approach consist of 3 steps as followed:
1. Selecting bus stops from all possible bus stops in the region by DBSCAN clustering based algorithm
2. Assigning customers to the selected bus stops using GRASP(Greedy Randomized Adaptive Search Procedure)
3. Finding optimal routes for the buses keeping in mind the constraints using VND(Variable Neighbourhood descent)

To be able to run the code, we will need the following data files:
* bus_stop_dist.csv : csv containg distance matrix of bus stops
* bus_stop_time.csv : csv containg time taken to reach one bus stop from another
* cust_stop_mat.csv : csv containg customer to all bus stops distance 
* bus_coord.npy : npy file containg all bus stop coordinates(latitude and longitude)
* cp_coord.npy : npy file containing all customer coordinate(latitude and longitude)

The code is tested with Python3 and requirements are listed in requirements.txt

To generate the output, run
```
python code.py
```

The outputs contains the assigment of cutomers to bus stops and the routes for each bus in array containg the index of the bus stop it has to stop at in temporal manner.

We have also provided a jupyter notebook file containing the same code. 
