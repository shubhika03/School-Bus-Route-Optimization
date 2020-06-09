import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import random
import csv
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from random import seed

def VRP (dist_mat, time_mat, cust_stop_mat, bs_points, cp_points, w_dist_lim, cap_lim, dist_lim, tim_lim):

    depot = bs_points[0]
    # Seleccting potential co-ordinates
    
    bs_points = bs_points[1:]
    points = cust_stop_mat

    cp_bs_dist = np.array(points[:,1:]).astype(np.float64)

    no_of_bs = cp_bs_dist.shape[1]
    no_of_cp = cp_bs_dist.shape[0]

    all_cust_index = np.array([i for i in range(no_of_cp)])   #for getting index of potelntial customer
    all_stop_index = np.array([i for i in range(no_of_bs)])
    cp_bs_assign = np.zeros(no_of_bs)
    extreme_necces = set()
    necessary_bs = np.ones(no_of_bs, dtype = bool)
    allocated_cp = set()

    for i in range(no_of_cp):
        
        curr_bs_dist = cp_bs_dist[i, :]
        possible_bs = (curr_bs_dist <= w_dist_lim)
        possible_bs_ind = all_stop_index[possible_bs]
        if np.sum(possible_bs) > 0:
            cp_bs_assign[possible_bs] += 1
            if(np.sum(possible_bs) > 1):
                flag = 0
                for st in possible_bs_ind:
                    if st in extreme_necces:
                        flag = 1
                        break
                        #continue
                    #necessary_bs[st] = False
                if flag == 0:
                    necessary_bs[possible_bs] = False                        
            else:
                necessary_bs[possible_bs_ind[0]] = True
                #print(possible_bs_ind[0])
                extreme_necces.add(possible_bs_ind[0])
                all_cust_dist = cp_bs_dist[:, possible_bs_ind[0]]
                reachable_cust = all_cust_dist <= w_dist_lim
                reachable_cust = list(all_cust_index[reachable_cust])
                allocated_cp.update(reachable_cust)

        else:
            possible_bs = np.argmin(curr_bs_dist)
            cp_bs_assign[possible_bs] += 1    
            necessary_bs[possible_bs] = True
            extreme_necces.add(possible_bs)
            all_cust_dist = cp_bs_dist[:, possible_bs]
            reachable_cust = all_cust_dist <= w_dist_lim
            reachable_cust = list(all_cust_index[reachable_cust])
            allocated_cp.update(reachable_cust)
 
            

    unnec_bs_points = bs_points[~necessary_bs]
    remove_bs_pts = cp_bs_assign == 0

    potential_bs = bs_points[(necessary_bs & (~remove_bs_pts))]
    

    cp_unnec_bs_dist = cp_bs_dist[:, ~necessary_bs]
    col = [0, 0, 0, 1]

    a = np.sqrt(np.sum((unnec_bs_points[0] - unnec_bs_points[1])**2))
    b = geodesic(unnec_bs_points[0], unnec_bs_points[1]).miles * 1.61
    scale = a/b * 0.5
    print(scale)

    db = DBSCAN(eps=scale, min_samples=2).fit(unnec_bs_points)

    labels = db.labels_
    unique_labels = set(labels)

    busstop_customer_alloc = []  #unnec bs to cutomer alloc


    for stop in range(unnec_bs_points.shape[0]):

        all_cust_dist = cp_unnec_bs_dist[:, stop]
        reachable_cust = all_cust_dist <= w_dist_lim
        reachable_cust = list(all_cust_index[reachable_cust])
        busstop_customer_alloc.append(reachable_cust)

    max_pts = unnec_bs_points.shape[0]

    tot_customer = set([i for i in range(no_of_cp)])
    unnec_bs_ind_seq = np.array([i for i in range(unnec_bs_points.shape[0])])

 

    for iter_no in range(5):

        curr_alloc_cp = set()
        curr_potential_bs = potential_bs.copy()

        for k in unique_labels:

            class_ind = (labels == k)
            curr_bs_pts = unnec_bs_points[class_ind]
            choose_no_pts = max(curr_bs_pts.shape[0]//3, 1)
            if k != -1:
               
                sequence = list(unnec_bs_ind_seq[class_ind])
                subset = random.sample(sequence, choose_no_pts)
                curr_potential_bs = np.vstack((curr_potential_bs, unnec_bs_points[subset]))
                for st in subset:
                    #print(busstop_customer_alloc[st])
                    curr_alloc_cp.update(busstop_customer_alloc[st])

          

        curr_alloc_cp = curr_alloc_cp.union(allocated_cp)
        unallocated_cp = tot_customer - curr_alloc_cp
    
        if(len(unallocated_cp) == 0):
            best_potential_bs = curr_potential_bs
            break    
        else:
            print("extra allocation")
            add_pts = 0
            for left_cp in unallocated_cp:
                if left_cp in curr_alloc_cp:
                    continue

                curr_cp_bs_dist = cp_unnec_bs_dist[left_cp, :]
                min_ind = np.argmin(curr_cp_bs_dist)
                curr_alloc_cp.update(busstop_customer_alloc[min_ind])
                curr_potential_bs = np.vstack((curr_potential_bs, unnec_bs_points[min_ind]))
                add_pts += 1

            if add_pts < max_pts:
                best_potential_bs = curr_potential_bs   

    list_ind = []
    for i in range(bus_coord.shape[0]):
        if bus_coord[i] in best_potential_bs:
            list_ind.append(i)

    if 0 not in list_ind:
        print ('Zero is inserted')
        list_ind.insert(0,0)

    # Removing duplicate points
    for i in range(len(list_ind)-1,-1,-1):
        for j in range(i-1,-1,-1):
            if list(bus_coord[list_ind[i]]) == list(bus_coord[list_ind[j]]):
                del list_ind[i]
 
    # Updating all the matrices
    dist_mat = dist_mat[list_ind,:]
    dist_mat = dist_mat[:,list_ind]
    time_mat = time_mat[list_ind,:]
    time_mat = time_mat[:,list_ind]
    cust_stop_mat = cust_stop_mat[:,list_ind]
    
    print (list_ind)
    def cal_tlk(cust_stop_mat, route, lim):
        num_rts = route.count(0) -1
        stops_in_r = [[] for i in range(num_rts)]
        j = 0
        for i in range(1, len(route)):
            if route[i] == 0:
                j += 1
            else:
                stops_in_r[j].append(int(route[i]))
        tlk = np.ones([len(cust_stop_mat), num_rts])
      
        for i in range(len(cust_stop_mat)):
            curr_cust_dist = cust_stop_mat[i][1:]
            if(min(curr_cust_dist) <= lim):

                for j in range(num_rts):
                    #print ([cust_stop_mat[i][k-1] for k in stops_in_r[j]])
                    if min([cust_stop_mat[i][k] for k in stops_in_r[j]]) <= lim:
                        tlk[i][j] = 0.000001
            else:
                nearest_bs = np.argmin(curr_cust_dist)+1
                #print (nearest_bs)
                for j in range(num_rts):
                    if nearest_bs in stops_in_r[j]:
                        tlk[i][j] = 0.000001
        return tlk

    def check_feasible(route, cust_stop_mat, w_dist_lim, cap_lim):
        tlk = cal_tlk(cust_stop_mat, route, w_dist_lim)
      
        num_cust = len(cust_stop_mat)
        num_rts = route.count(0) - 1
       

        x = pulp.LpVariable.dicts('route_allocation', [(i,j) for i in range(num_cust) for j in range(num_rts)], cat = 'Binary')
        allocation = pulp.LpProblem("Allocation Model", pulp.LpMinimize)

        # Objective function
        allocation += pulp.lpSum([x[i,j]*tlk[i][j] for i in range(num_cust) for j in range(num_rts)])

        # Constraints
        for i in range(num_cust):
            allocation += pulp.lpSum(x[i,j] for j in range(num_rts)) == 1

        for i in range(num_rts):
             allocation += pulp.lpSum(x[j,i] for j in range(num_cust)) <= cap_lim

        for i in range(num_cust):
            for j in range(num_rts):
                allocation += x[i,j]*tlk[i][j] <= 0.1
  

        allocation.solve()

        matt = []
        for i in range(num_cust):
            curr_alloc = []
            for j in range(num_rts):
                curr_alloc.append(x[i,j].varValue)
            matt.append(curr_alloc)   

        if pulp.LpStatus[allocation.status] == 'Optimal':
            return 0, matt
        return 1, matt

    def route_update(array, m_stops):
        ind_slice = []
        temp = [0,0]
        # base condition


        for i in range(len(array)):
            if array[i] in m_stops:
                ind_slice.append((i - array[i::-1].index(0), i+ array[i:].index(0)))
        slice1 = []
        slice2 = []

        if len(set(ind_slice)) != 2:
            return array,list(map(int,m_stops))
        if m_stops[0] == array[ind_slice[0][1]-1]:
            slice1 = array[ind_slice[0][0]+1:ind_slice[0][1]].copy()
            slice2 = array[ind_slice[1][0]+1:ind_slice[1][1]].copy()  
    
        elif m_stops[0] == array[ind_slice[1][1]-1]:
            slice1 = array[ind_slice[1][0]+1:ind_slice[1][1]].copy()
            slice2 = array[ind_slice[0][0]+1:ind_slice[0][1]].copy()
          
        else:
            return array,list(map(int,m_stops))

        del array[ind_slice[1][0]:ind_slice[1][1]], array[ind_slice[0][0]:ind_slice[0][1]]
        array = array + slice1 + slice2 + [0]
        return array,list(map(int,m_stops))

    def CnW_matrix(dist_mat, n):
        savings_matrix = np.zeros([n-1,n-1])
        for i in range(1,n):
            for j in range(1,n):
                if i==j:
                    savings_matrix[i-1][j-1] = 0
                else:
                    d1 = dist_mat[i][0]
                    d2 = dist_mat[j][0]
                    d = dist_mat[i][j]
                    if (d1 + d2 - d >= 0):
                        savings_matrix[i-1][j-1] = d1 + d2 - d
        return savings_matrix

    def roulette_wheel(prob):
        num = len(prob)
        cumm_prob = np.zeros([num, num])
        prob_cumm = 0
        for i in range(num):
            for j in range(num):
                cumm_prob[i][j] = prob_cumm
                prob_cumm = prob_cumm + prob[i][j]
        r = np.random.rand()
        for i in range(num):
            for j in range(num):
                if j!= num-1 :
                    if r < cumm_prob[i][j+1]:
                        if r >= cumm_prob[i][j]:
                            return (i+1,j+1)
                else:
                    if i != num-1 :
                        if r < cumm_prob[i+1][0]:
                            if r >= cumm_prob[i][j]:
                                return (i+1,j+1)
                    else:
                        return (num, num)

    def check_tim_dist(route, dis_lim, tim_lim, dist_mat, tim_mat):
        s_tim = 0
        s_dist = 0
        for i in range(1, len(route)-1):
            if route[i] != 0:
                s_tim += tim_mat[route[i]][route[i+1]]
                s_dist += dist_mat[route[i]][route[i+1]]
            else:
                if s_tim >= tim_lim:
                    return False
                if s_dist >= dist_lim:
                    return False
                s_tim = 0
                s_dist = 0
        return True

    def check_feasible_vnd(route, cust_stop_mat, w_dist_lim, cap_lim, min_cap):
        tlk = cal_tlk(cust_stop_mat, route, w_dist_lim)
   
        num_cust = len(cust_stop_mat)
        num_rts = route.count(0) - 1
      

        x = pulp.LpVariable.dicts('route_allocation', [(i,j) for i in range(num_cust) for j in range(num_rts)], cat = 'Binary')
      
        allocation = pulp.LpProblem("Allocation Model", pulp.LpMinimize)

        # Objective function
        allocation += pulp.lpSum([x[i,j]*tlk[i][j] for i in range(num_cust) for j in range(num_rts)]) 

        for i in range(num_cust):    
            allocation += pulp.lpSum(x[i,j] for j in range(num_rts)) == 1

        for i in range(num_rts):
            allocation += pulp.lpSum(x[j,i] for j in range(num_cust)) <= cap_lim

        for i in range(num_rts):
            allocation += pulp.lpSum(x[j,i] for j in range(num_cust)) >= min_cap*cap_lim

        for i in range(num_cust):
            for j in range(num_rts):
                allocation += x[i,j]*tlk[i][j] <= 0.1
 

        allocation.solve()
        matt = []
        for i in range(num_cust):
            curr_alloc = []
            for j in range(num_rts):
                curr_alloc.append(x[i,j].varValue)
            matt.append(curr_alloc)   


        if pulp.LpStatus[allocation.status] == 'Optimal':
            return 0, matt
        return 1, matt


    # remove and insert within route
    def rem_ins_w_rt(array):
        rand1 = np.random.randint(len(array))         # generates random node
        while (array[rand1] == 0):    
            rand1 = np.random.randint(len(array))
  
        s = 0
        e = 0
        for i in range(len(array)):
            if (array[i] == 0) and (i < rand1):
                s = i
                continue
            if (array[i] == 0) and (i > rand1):
                e = i
                break
      
        rand2 = s + np.random.randint(e-s-1)+1
        val = array[rand1]
        del array[rand1]
        array.insert(rand2, val)
        array = list(map(int,array))
        for i in range(len(array)-1,0,-1):
            if array[i] == array[i-1]:
                del array[i]
        return array

    # remove insert between different routes
    def rem_ins_bwt_rt(array):
        r1 = np.random.randint(len(array))   # generates two distinct nodes and then swaps those nodes
        while (array[r1] == 0):
            r1 = np.random.randint(len(array))
        r2 = np.random.randint(len(array))
        while ((array[r2] == 0) or (r1 == r2)):
            r2 = np.random.randint(len(array))
      
        array.insert(r2, array.pop(r1))
        array = list(map(int,array))
        for i in range(len(array)-1,0,-1):
           
            if array[i] == array[i-1]:
                del array[i]
    
        return array

    # replace node with an unallocated bus stop
    def replace_nodes(array, pot_stops_coord, coord):
        r1 = np.random.randint(len(array))
        while (array[r1] == 0):
            r1 = np.random.randint(len(array))
        # Finding nearest stop for array[r1]
        min1 = np.inf
        stop = 0
        for i in range(len(pot_stops)):
            if i in array:
                continue
            a = euclidean_distance(pot_stops_coord[array[r1]], pot_stops_coord[i])
            if (min1 > a):
                min1 = a
                stop = i
        array[r1] = i
        return array

    def remove_node(array):
        r1 = np.random.randint(len(array))
        while (array[r1] == 0):
            r1 = np.random.randint(len(array))
      
        del array[r1]
        array = list(map(int,array))
        for i in range(len(array)-1,0,-1):
         
            if array[i] == array[i-1]:
                del array[i]
        return array

    def cal_cost(array, dist_mat):
        cost = 0
        for i in range(1,len(array)-1):
            if array[i] == 0:
                continue
            cost += dist_mat[array[i]][array[i+1]]
        return cost
    min_vehicles = dist_mat.shape[0]
    for i in range(5):
        print('GRASP Initialised', i)
        # Running the GRASP Algorithm
        num_stops = dist_mat.shape[0]
        route = np.zeros([2*num_stops-1])

        #Initial Sol for Clarke and Wright
        for i in range(1,len(route)):
            if (i-1)%2 == 0:
                route[i] = int(i/2) + 1

        savings_matrix = CnW_matrix(dist_mat, num_stops)
        list_stop_pairs = []
        for i in range(num_stops-1):
            for j in range(num_stops-1):
                if i != j:
                    l = []
                    l.append(i+1)
                    l.append(j+1)
                    list_stop_pairs.append(l)
        prob_sum = 1
        while (len(list_stop_pairs) >=0 and (prob_sum >= 0.1)):
            prob_sum = 0.0000000001
            prob = np.zeros([num_stops-1, num_stops-1])
            for k in range(len(list_stop_pairs)):
                i = list_stop_pairs[k][0]
                j = list_stop_pairs[k][1]
                prob_sum = prob_sum + savings_matrix[i-1][j-1]
            for k in range(len(list_stop_pairs)):
                i = list_stop_pairs[k][0]
                j = list_stop_pairs[k][1]
                prob[i-1][j-1] = savings_matrix[i-1][j-1]/prob_sum

            stop = roulette_wheel(prob)
            upd_route, temp = route_update(list(map(int,route.copy())), stop)

            obj_val, cust_alloc = check_feasible(list(upd_route), cust_stop_mat, w_dist_lim, cap_lim)
            if (obj_val <= 0.1) and (check_tim_dist(upd_route, dist_lim, tim_lim,dist_mat, time_mat)):
                route = upd_route.copy()
                cust_alloc_true = cust_alloc.copy()
                prob[temp[0]-1] = np.zeros(num_stops-1)
                prob[:][temp[1]-1] = np.zeros(num_stops-1).T
                for m in range(len(list_stop_pairs)-1,-1,-1):
                    if list_stop_pairs[m][0] == temp[0]:
                        del list_stop_pairs[m]
                    elif list_stop_pairs[m][1] == temp[1]:
                        del list_stop_pairs[m]
            else:
                list_stop_pairs.remove(list(stop))
        

        # max vehicles
        route = list(map(int,route))
        for i in range(len(route)-1,0,-1):
        
            if route[i] == route[i-1]:
                del route[i]
        num_veh = route.count(0)-1
        if num_veh < min_vehicles:
            min_vehicles_route = route
            min_vehicles = num_veh
        if min_vehicles == 1:
            break
    print('GRASP Phase Completed')
    it_dist = []
    route = min_vehicles_route
    dist = 0
    for i in range(len(route)-1):
        if route[i] == 0:
            continue
        else:
            dist += dist_mat[route[i]][route[i+1]]
    it_dist.append(dist)
    # VND

    # array - encoded route
    # pot_stops - coordinates of all stops
    #def VND(array, pot_stops, dist_mat):
    print (route)
    print('VND Initialised')
    l = 0.5
    lmax = 0.91
    if route.count(0)-1 == 1:
        l = (cust_stop_mat.shape[0]/cap_lim) - 0.05
        lmax = l + 0.05
    cnt = 0
    check = 0
    array = route.copy()
    for i in range(5):
        K_max = 200
        print (l)
        l_opt = l
   
        K = 0
        cnt = 0
        flag = 0
        array = route.copy()
        best_sol = array.copy()
        n_array = [array.copy() for i in range(4)]
        cost_best_sol = cal_cost(best_sol, dist_mat)
        while (K<K_max):
            dist = 0
            cost = [0 for i in range(4)]
            # generate neighbourhood

         
            n_array[0] = rem_ins_w_rt(array.copy())
          
            n_array[1] = rem_ins_bwt_rt(array.copy())
          
            n_array[2] = remove_node(array.copy())
            
            # Applying al 3 operations on the array and storing in 4th variable
            n_array[3] = rem_ins_w_rt(array.copy())
            n_array[3] = rem_ins_bwt_rt(n_array[3].copy())
            n_array[3] = remove_node(n_array[3].copy())

            l = round(l, 1)
            for i in range(4):
            
                obj_val, cust_alloc = check_feasible_vnd(list(map(int,n_array[i])), cust_stop_mat, w_dist_lim, cap_lim, l)
                if (obj_val >0.1) or (check_tim_dist(list(map(int,n_array[i])), dist_lim, tim_lim,dist_mat, time_mat) == False):
                    n_array[i] = np.inf
                else:
                    cust_alloc_vnd = cust_alloc.copy()

            for i in range(4):
                if n_array[i] != np.inf :
                    cost[i] = cal_cost(n_array[i], dist_mat)
                else:
                    cost[i] = np.inf        

            ind = cost.index(min(cost))
            if min(cost) < cost_best_sol :
                cost_best_sol = min(cost)
                best_sol = n_array[ind].copy()
                array = best_sol.copy()
                cnt += 1
                K = 1
            else:
                K += 1 
            dist = 0
            for i in range(len(best_sol)-1):
                if best_sol[i] == 0:
                    continue
                else:
                    dist += dist_mat[best_sol[i]][best_sol[i+1]]
            it_dist.append(dist)

        
        if (check == 1): 
            if (cnt == 0):
                break
            else:
                if l < lmax:
                    l += 0.1
                else:
                    break
        elif (check == -1) :
            if (cnt > 0):
                break
            else:
                l -= 0.1
        else:
            if cnt > 0:
                l += 0.1
                check = 1
            else:
                l -= 0.1
                check = -1

    print (l_opt, route,'\n', best_sol)
    return list_ind, route, best_sol, cust_alloc_vnd, l



dist_mat = pd.read_csv('bus_stop_dist.csv')
dist_mat.drop('Unnamed: 0', axis = 1, inplace = True)
time_mat = pd.read_csv('bus_stop_time.csv')
time_mat.drop('Unnamed: 0', axis = 1, inplace = True)
cust_stop_mat = pd.read_csv('cust_stop_mat.csv')
cust_stop_mat = np.asarray(cust_stop_mat)
cust_stop_mat = cust_stop_mat[1:]
#cust_stop_mat = cust_stop_mat[:,1:]
dist_mat = np.asarray(dist_mat)
time_mat = np.asarray(time_mat)
bus_coord = np.load('bs_coord.npy')
cp_coord = np.genfromtxt("cust_coord.csv", delimiter=";", skip_header=1)

import timeit

start = timeit.default_timer()

# Parameters
# dist_mat : bus stop distance matrix
# time_mat : bus stop time matrix
# cust_stop_mat : customer to bus stop distance matrix
# bs_points : bus stop coordinates
# cp_points : customer coordinates
# w_dist_lim : walking distance limit
# cap_lim : capacity limit
# dist_lim : total distance limit
# time_lim : total time limit
list_ind, route, best_sol, cust_alloc_vnd, l = VRP(dist_mat, time_mat, cust_stop_mat, bus_coord, cp_coord,1000,10,60000,6000)

stop = timeit.default_timer()

print('Time: ', stop - start)