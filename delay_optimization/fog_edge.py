#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 06:27:42 2022

@author: user
"""

from math import ceil

from numpy import exp
from numpy.random import rand
from numpy.random import seed

from random_partitioning import generate, generate_with_lower_bounds
import parameters as p
import pandas as pd

speedOfLight = 299792458
# some small value
eps = 0.00001
# minimum_traffic_threshold = 1000  # TODO MJ: rethink this value
available_architectures = list(i + 1 for i in range(10))
available_resources = ['F1', 'F2', 'F3', 'E1', 'E2', 'E3', 'C1', 'C2', 'C3']
F1, F2, F3, E1, E2, E3, C1, C2, C3 = available_resources
architecture_resource_mapping = {
    1: [F1, F2, F3],
    2: [E1, E2, E3],
    3: [C1, C2, C3],
    4: [F1, F2, E3],
    5: [F1, F2, C3],
    6: [E1, E2, C3],
    7: [F1, E2, E3],
    8: [F1, C2, C3],
    9: [E1, E2, C3],
    10: [F1, E2, C3],
}
# =============================================================================
# INPUT VARIABLES
# =============================================================================
N = p.N
K = p.K

traffic = p.traffic

miuW1 = p.miuW1 #1083441

# # ##DT+DNN
miuW2 = p.miuW2
miuW3 = p.miuW3

# # DNN
# miuW2 = 908214
# miuW3 = 915083

##DT
# miuW2 = 114549
# miuW3 = 114849

# # # RF
# miuW2 = 153288
# miuW3 = 167493

# #GB
# miuW2 = 121488
# miuW3 = 197266

# MLP
# miuW2 = 304367
# miuW3 = 312614

S = p.S

Sf = p.Sf
Se = p.Se
Sc = p.Sc

miuL1 = p.miuL1  # in kilobits
miuL2 = p.miuL2
miuL3 = p.miuL3

pA = p.pA

CF = (S / Sf) / (K * N)
CE = (S / Se) / N
CC = (S / Sc)

CUF = 1e6
CFE1 = 2.5e7 * 4
CFE2 = 2.5e7 * 4
CFE3 = 2.5e7 * 4
CEC1 = 2.5e7 * 4
CEC2 = 2.5e7 * 4
CEC3 = 2.5e7 * 4

distanceUF = 1000  # in Meters
dUF = distanceUF / speedOfLight

distanceFE = 10000
dFE = distanceFE / speedOfLight

distanceEC = 1000000
dEC = distanceEC / speedOfLight

seed(1)

temp = 1000


# =============================================================================
# FORMULA FUNCTION TO CALCULATE THE DELAY
# =============================================================================
def formula(pj, rC):
    ### TRAFFIC OFFLOADING RATE ###
    p = [p_ * traffic for p_ in pj]
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = p

    ### CALCULATE CAPACITY FROM GIVEN COSTS ###
    CF1 = (((S / Sf) * rC[0]) / (K * N)) / miuW1
    CF2 = (((S / Sf) * rC[1]) / (K * N)) / miuW2
    CF3 = (((S / Sf) * rC[2]) / (K * N)) / miuW3

    CE1 = (((S / Se) * rC[3]) / N) / miuW1
    CE2 = (((S / Se) * rC[4]) / N) / miuW2
    CE3 = (((S / Se) * rC[5]) / N) / miuW3

    CC1 = (((S / Sc) * rC[6]) / 1) / miuW1
    CC2 = (((S / Sc) * rC[7]) / 1) / miuW2
    CC3 = (((S / Sc) * rC[8]) / 1) / miuW3

    ### CALCULATE ARRIVAL RATE FOR EACH TIER ###
    lambdaF1 = p1 + p4 + p5 + p7 + p8 + p10
    lambdaF2 = p1 + p4 + p5
    lambdaF3 = p1

    lambdaFE1 = K * (p2 + p3 + p6 + p9)
    lambdaFE2 = K * (p7 + p8 + p10)
    lambdaFE3 = K * (p1)

    lambdaE1 = K * (p2 + p6 + p9)
    lambdaE2 = K * (p2 + p6 + p7 + p10)
    lambdaE3 = K * (p2 + p4 + p7)

    lambdaEC1 = N * K * (p3)
    lambdaEC2 = N * K * (p8 + p9)
    lambdaEC3 = N * K * (p5 + p6 + p10)

    lambdaC1 = N * K * (p3)
    lambdaC2 = N * K * (p3 + p8 + p9)
    lambdaC3 = N * K * (p3 + p5 + p6 + p8 + p9 + p10)

    ### CALCULATE DELAY IN EACH TIER AND LINK ###
    DUF = (1 / (CUF - traffic)) + dUF
    
    DF1 = 0 if CF1 <= 0 else 1 / (CF1 - lambdaF1)
    if DF1 < 0: DF1 = 1000
    DF2 = 0 if CF2 <= 0 else 1 / (CF2 - lambdaF2)
    if DF2 < 0: DF2 = 1000
    DF3 = 0 if CF3 <= 0 else 1 / (CF3 - (lambdaF3 * pA))
    if DF3 < 0: DF3 = 1000

    DFE1 = ((1 / (CFE1 - (lambdaFE1)))) + dFE
    DFE2 = ((1 / (CFE2 - (lambdaFE2)))) + dFE
    DFE3 = ((1 / (CFE3 - (lambdaFE3)))) + dFE

    DE1 = 0 if CE1 <= 0 else 1 / (CE1 - lambdaE1)
    if DE1 < 0: DE1 = 1000
    DE2 = 0 if CE2 <= 0 else 1 / (CE2 - lambdaE2)
    if DE2 < 0: DE2 = 1000
    DE3 = 0 if CE3 <= 0 else 1 / (CE3 - (lambdaE3 * pA))
    if DE3 < 0: DE2 = 1000

    DEC1 = ((1 / (CEC1 - (lambdaEC1)))) + dEC
    DEC2 = ((1 / (CEC2 - (lambdaEC2)))) + dEC
    DEC3 = ((1 / (CEC3 - (lambdaEC3)))) + dEC

    DC1 = 0 if CC1 <= 0 else 1 / (CC1 - lambdaC1)
    if DC1 < 0: DC1 = 1000
    DC2 = 0 if CC2 <= 0 else 1 / (CC2 - lambdaC2)
    if DC2 < 0: DC2 = 1000
    DC3 = 0 if CC3 <= 0 else 1 / (CC3 - (lambdaC3 * pA))
    if DC3 < 0: DC3 = 1000

    D1 = DUF + DF1 + DF2 + DF3
    D2 = DUF + DFE1 + DE1 + DE2 + DE3
    D3 = DUF + DFE1 + DEC1 + DC1 + DC2 + DC3
    D4 = DUF + DFE3 + DF1 + DF2 + DE3
    D5 = DUF + DF1 + DF2 + DFE3 + DEC3 + DC3
    D6 = DUF + DFE1 + DE1 + DE2 + DEC3 + DC3
    D7 = DUF + DFE2 + DF1 + DE2 + DE3
    D8 = DUF + DF1 + DFE2 + DEC2 + DC2 + DC3
    D9 = DUF + DFE1 + DE1 + DEC2 + DC2 + DC3
    D10 = DUF + DF1 + DFE2 + DF2 + DEC3 + DC3

    Dj = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10]

    # NOTE: there's no need to divide by traffic
    D = sum(d_ * p_ for d_, p_ in zip(Dj, pj))

    return D


# =============================================================================
# OPTIMIZATION FUNCTION FOR OPTIMIZING COST ALLOCATION
# =============================================================================
def optimize_cost(n_iterations, temp, pj, on_resources):
    # generate an initial point
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [p_ * traffic for p_ in pj]
    # print(pj)

    # NOTE: pA is added to calculations later!
    lambdaF1 = p1 + p4 + p5 + p7 + p8 + p10
    lambdaF2 = p1 + p4 + p5
    lambdaF3 = p1
    lambdaE1 = K * (p2 + p6 + p9)
    lambdaE2 = K * (p2 + p6 + p7 + p10)
    lambdaE3 = K * (p2 + p4 + p7)
    lambdaC1 = N * K * (p3)
    lambdaC2 = N * K * (p3 + p8 + p9)
    lambdaC3 = N * K * (p3 + p5 + p6 + p8 + p9 + p10)

    lambdas = lambdaF1, lambdaF2, lambdaF3, lambdaE1, lambdaE2, lambdaE3, lambdaC1, lambdaC2, lambdaC3
    # print(lambdas)
    # resources = [r for r, l in zip(available_resources, lambdas) if l >= minimum_traffic_threshold]
    # print(resources)

    ### CALCULATE MINIMUM COST FOR EACH TIER --> TO AVOID MINIMUM DELAY ###
    # NOTE: the K and N refer to the fact that the minimum cost is multiplied by the number of instances in the system
    minSF1 = 0 if lambdaF1 <= 0 else ceil(K * N * lambdaF1 * miuW1 * Sf)
    minSF2 = 0 if lambdaF2 <= 0 else ceil(K * N * lambdaF2 * miuW2 * Sf)
    minSF3 = 0 if lambdaF3 <= 0 else ceil(K * N * pA * lambdaF3 * miuW3 * Sf)
    minSE1 = 0 if lambdaE1 <= 0 else ceil(N * lambdaE1 * miuW1 * Se)
    minSE2 = 0 if lambdaE2 <= 0 else ceil(N * lambdaE2 * miuW2 * Se)
    minSE3 = 0 if lambdaE3 <= 0 else ceil(N * pA * lambdaE3 * miuW3 * Se)
    minSC1 = 0 if lambdaC1 <= 0 else ceil(1 * lambdaC1 * miuW1 * Sc)
    minSC2 = 0 if lambdaC2 <= 0 else ceil(1 * lambdaC2 * miuW2 * Sc)
    minSC3 = 0 if lambdaC3 <= 0 else ceil(1 * pA * lambdaC3 * miuW3 * Sc)

    minimums = minSF1, minSF2, minSF3, minSE1, minSE2, minSE3, minSC1, minSC2, minSC3

    lower_bounds = {r: m for r, m in zip(available_resources, minimums) if r in on_resources}
    random_spending = generate_with_lower_bounds(S, lower_bounds)
    best_spending = [random_spending.get(r, 0) for r in available_resources]
    # best_spending = minimums

    # evaluate the initial point
    best_eval_D = formula(pj, best_spending)

    # current working solution
    curr_spending, curr_eval_D = best_spending, best_eval_D
    cost_optimization_history = list()
    for i in range(n_iterations):
        random_spending = generate_with_lower_bounds(S, lower_bounds)
        candidate_spending = [random_spending.get(r, 0) for r in available_resources]
        candidate_eval_D = formula(pj, candidate_spending)
        if candidate_eval_D < best_eval_D:
            best_spending, best_eval_D = candidate_spending, candidate_eval_D
            cost_optimization_history.append(best_eval_D)

        diff = candidate_eval_D - curr_eval_D
        t = temp / float(i + 1)
        metropolis = exp(-diff / t)
        if diff < 0 or rand() < metropolis:
            curr_spending, curr_eval_D = candidate_spending, candidate_eval_D

    # NOTE MJ: this returns a normalized spending
    return [bs / S for bs in best_spending]


# =============================================================================
# OPTIMIZATION FUNCTION FOR OPTIMIZING OFFLOADING RATIOS
# =============================================================================
def optimize_offloading(n_iterations, n_cost_iterations, temp, on_architectures, on_resources):
    # generate an initial point
    best_pj = [round(1 / len(on_architectures), 3) if j in on_architectures else 0 for j in available_architectures]

    ### RUN COST ALLOCATION OPTIMIZATION TO GET THE OPTIMUM COST FOR EACH OFFLOADING SOLUTION ###
    best_rC = optimize_cost(n_cost_iterations, temp, best_pj, on_resources)

    best_eval_D = formula(best_pj, best_rC)

    curr_pj, curr_eval_D = best_pj, best_eval_D
    curr_rC = best_rC
    offloading_optimization_history = []
    for i in range(n_iterations):
        normalizedTotalOffloading = 1
        candidate_pj = generate(normalizedTotalOffloading, on_architectures, available_architectures)

        ### RUN COST ALLOCATION OPTIMIZATION TO GET THE OPTIMUM COST FOR EACH OFFLOADING SOLUTION ###
        cand_rC = optimize_cost(p.n_iterations_cost, temp, candidate_pj, on_resources)
        # print(rC)
        
        candidate_eval_D = formula(candidate_pj, cand_rC)

        if candidate_eval_D < best_eval_D:
            best_pj, best_eval_D = candidate_pj, candidate_eval_D
            best_rC = cand_rC
            offloading_optimization_history.append(best_eval_D)
            print(f'Result for {i}:\tdelay {round(best_eval_D * 1000, 3)} ms\t| offloading {best_pj}\t')
            # print(rC)

        # NOTE: curr_pj isn't used anywhere and we're performing a simple Random Search here
        diff = candidate_eval_D - curr_eval_D
        t = temp / float(i + 1)
        metropolis = exp(-diff / t)
        if diff < 0 or rand() < metropolis:
            # print('metropolis')
            curr_pj, curr_eval_D = candidate_pj, candidate_eval_D
            curr_rC = cand_rC

    return [best_pj, best_rC, best_eval_D, offloading_optimization_history]

    # =============================================================================
    # START THE OPTIMIZATION
    # =============================================================================


if __name__ == '__main__':
    
    for x in range(6):
        print('Iteration: ', x)
        # fog_edge - means the optimization is done for 1, 2, 4 and 7
        fog_edge_architectures = [1, 2, 4, 7]
        fog_edge_resources = [F1, F2, F3, E1, E2, E3]
        
        # fog_cloud_architectures = [1, 3, 5, 8]
        # fog_cloud_resources = [F1, F2, F3, C1, C2, C3]
        
        pj, rC, best_eval_D, scores = optimize_offloading(p.n_iterations, p.n_iterations_cost, temp,
                                                          fog_edge_architectures, fog_edge_resources)
        pct_rC = [round(r * 100, 3) for r in rC]
        print(f'Best delay {best_eval_D * 1000:.2f} ms obtained for offloading:\n\t{pj}\nand cost allocation\n\t{pct_rC}')
        for r, prC in zip(available_resources, pct_rC):
            print(f'{r}: {prC}')
        for (j, rs), _pj in zip(architecture_resource_mapping.items(), pj):
            print(f'{j}:\t{rs} - {_pj * 100:.2f}%')
            
        print('')
        print('Results to table')
        d = [['fog-edge', pj[0], pj[1], pj[2], pj[3], pj[4], pj[5], pj[6], pj[7], pj[8], pj[9],
              pct_rC[0], pct_rC[1],pct_rC[2],pct_rC[3],pct_rC[4],pct_rC[5],pct_rC[6],pct_rC[7],pct_rC[8],best_eval_D * 1000]]
    
        df = pd.DataFrame(d, columns = ['arch','t1','t2','t3', 't4', 't5','t6', 't7','t8','t9','t10',
                                        'F1', 'F2','F3','E1', 'E2', 'E3', 'C1', 'C2', 'C3','Delay'])
        print(df)
        df.to_csv('results.csv', mode='a', header=False)
        
    
    # NOTE: maybe the input total cost is too much, we use only 6% and we don't care about the rest
    # TODO: consider only meaningful resources?
    # TODO: try to operate on microseconds or even less?
    # TODO: try to go below 3 ms on some resource - force it
    # TODO: sprawdzić czy przesunięcie kosztu z nieużywanego na używane poprawi wynik przy konkretnym rozwiązaniu
