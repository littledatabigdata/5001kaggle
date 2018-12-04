# SGD Classifier parameters #
# penalty - regularization scheme, l2, l1, or enet, enet=L2+L1, D=l2, only nominal feature
# l1_ratio - The Elastic Net mixing parameter  0 <= l1_ratio <= 1. 
# l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15
# alpha - constant that multiplies the regularization term Defaults to 0.0001

# max_iter - The maximum number of passes over the training data (aka epochs). higher should mean slower
# random_state - seed of RNG, drop this shit
# n_jobs - The number of CPUs to use to do the OVA computation. -1 means using all processors. max 8 cores

# data generation parameters #
# n_samples - number of samples D=100, all in mid-high hundreds
# n_classes - number of classes D=2, 2-10
# n_clusters_per_class - number of clusters per class D=2, 2-5
# n_informative - useful features, around 10
# flip_y - ratio of samples with wrong label D=0.01
# scale - feature scaling D=1, if none, then random scaled, 10-100, should be useless