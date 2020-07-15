import numpy as np

num_classes = 96

gate_weights = np.zeros((1, num_classes))
gate_lower_bound = 1
gate_upper_bound = 6
gate_weights[:, gate_lower_bound:gate_upper_bound] = 1
# gate_weights = np.reshape(gate_weights, (1, 57))

trn_weights = np.zeros((1, num_classes))
trn_lower_bound = 6
trn_upper_bound = 30
trn_weights[:, trn_lower_bound:trn_upper_bound] = 1

veh_weights = np.zeros((1, num_classes))
veh_lower_bound = 30
veh_upper_bound = 66
veh_weights[:, veh_lower_bound:veh_upper_bound] = 1

ped_weights = np.zeros((1, num_classes))
ped_lower_bound = 66
ped_upper_bound = num_classes
ped_weights[:, ped_lower_bound:ped_upper_bound] = 1

ped_in_crossing_weights = np.zeros((1, num_classes))
ped_in_crossing_lower_bound = 66
ped_in_crossing_upper_bound = 70
ped_in_crossing_weights[:, ped_in_crossing_lower_bound:ped_in_crossing_upper_bound] = 1
ped_in_crossing_lower_bound = 78
ped_in_crossing_upper_bound = num_classes
ped_in_crossing_weights[:, ped_in_crossing_lower_bound:ped_in_crossing_upper_bound] = 1

ped_not_in_crossing_weights = np.zeros((1, num_classes))
ped_not_in_crossing_lower_bound = 70
ped_not_in_crossing_upper_bound = 78
ped_not_in_crossing_weights[:, ped_not_in_crossing_lower_bound:ped_not_in_crossing_upper_bound] = 1

metric_weights = {
  'gate': gate_weights,
  'trn': trn_weights,
  'veh': veh_weights,
  'ped': ped_weights,
  'ped_in_crossing': ped_in_crossing_weights,
  'ped_not_in_crossing': ped_not_in_crossing_weights
}

weight_bounds = {
  'gate': (gate_lower_bound, gate_upper_bound),
  'trn': (trn_lower_bound, trn_upper_bound),
  'veh': (veh_lower_bound, veh_upper_bound),
  'ped': (ped_lower_bound, ped_upper_bound)
}