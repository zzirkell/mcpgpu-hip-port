# Integration Phase: Identified Problems

This document collects and describes critical issues identified during the integration phase of porting the MPCGPU project from NVIDIA CUDA to the AMD HIP backend.

---

## Problem 1: Strict C++ Bounds Checking and Post-Processing Crashes

### Description
The full simulation crashes with a `std::out_of_range` (`vector::_M_range_check`) error or a segmentation fault at the very end of its execution sequence. 

Code snippet:

./examples/pcg.exe 
Knot points: 32
State size: 14
Datatype: FLOAT
Sqp exits condition: CONSTANT TIME
QD COST: 0.0001
R COST: 0.0001
Rho factor: 1.2
Rho max: 10
Test iters: 1
Max sqp time: 2000
Solver: PCG
Max pcg iter: 173
Save data: OFF
Jitters: ON


start: 0 goal: 0
Logging test results to files with prefix tmp/results/32_PCG_0.000005 
Completed at 20260616_014012

RESULTS*************************************
Exit tol: 5e-06

Tracking err
Average[1.28045] Std Dev [0.537483] Min [0.000757515] Max [2.44134] Median [1.28781] Q1 [0.921525] Q3 [1.71158]
Average final tracking err: 2.02997

terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
Aborted (core dumped)

---

## Problem 2: Numerical Divergence in the SQP / PCG Loop

### Description
Although the complete ported stack (comprising the NMPC wrapper, generated robot dynamics, and the parallel linear system solver) compiles cleanly and executes without any memory access errors or crashes, the final mathematical tracking results diverge significantly between the NVIDIA CUDA baseline and the AMD HIP execution.

nvidia cuda: Tracking err
Average[0.109047] Std Dev [0.0863857] Min [0.000757456] Max [0.434672] Median [0.0957153] Q1 [0.0359972] Q3 [0.162921]
Average final tracking err: 0.0570047

amd hip: Tracking err
Average[1.28045] Std Dev [0.537483] Min [0.000757515] Max [2.44134] Median [1.28781] Q1 [0.921525] Q3 [1.71158]
Average final tracking err: 2.02997

==> min is good, the rest diverges on AMD ==> not good, need to find out where the error is