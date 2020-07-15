import numpy as np

gate_weights = np.zeros((1, 204))
gate_lower_bound = 1
gate_upper_bound = 6
gate_weights[:, gate_lower_bound:gate_upper_bound] = 1
# gate_weights = np.reshape(gate_weights, (1, 204))

trn_weights = np.zeros((1, 204))
trn_lower_bound = 6
trn_upper_bound = 30
trn_weights[:, trn_lower_bound:trn_upper_bound] = 1

veh_weights = np.zeros((1, 204))
veh_lower_bound = 30
veh_upper_bound = 66
veh_weights[:, veh_lower_bound:veh_upper_bound] = 1

cyc_weights = np.zeros((1, 204))
cyc_lower_bound = 66
cyc_upper_bound = 132
cyc_weights[:, cyc_lower_bound:cyc_upper_bound] = 1

ped_weights = np.zeros((1, 204))
ped_lower_bound = 132
ped_upper_bound = 204
ped_weights[:, ped_lower_bound:ped_upper_bound] = 1

metric_weights = {
  'gate': gate_weights,
  'trn': trn_weights,
  'veh': veh_weights,
  'cyc': cyc_weights,
  'ped': ped_weights
}

weight_bounds = {
  'gate': (gate_lower_bound, gate_upper_bound),
  'trn': (trn_lower_bound, trn_upper_bound),
  'veh': (veh_lower_bound, veh_upper_bound),
  'cyc': (cyc_lower_bound, cyc_upper_bound),
  'ped': (ped_lower_bound, ped_upper_bound)
}