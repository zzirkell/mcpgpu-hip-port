First step on both:

nvidia cuda: Tracking err
Average[0.109047] Std Dev [0.0863857] Min [0.000757456] Max [0.434672] Median [0.0957153] Q1 [0.0359972] Q3 [0.162921]
Average final tracking err: 0.0570047

amd hip: Tracking err
Average[1.28045] Std Dev [0.537483] Min [0.000757515] Max [2.44134] Median [1.28781] Q1 [0.921525] Q3 [1.71158]
Average final tracking err: 2.02997


==> min is good, the rest diverges on AMD ==> not good, need to find out where the error is

more time for amd so it has time to converge