import numpy as np

gate_weights = np.zeros((1, 204))
gate_weights[:, 1:6] = 1
# gate_weights = np.reshape(gate_weights, (1, 204))

trn_weights = np.zeros((1, 204))
trn_weights[:, 6:30] = 1

veh_weights = np.zeros((1, 204))
veh_weights[:, 30:66] = 1

cyc_weights = np.zeros((1, 204))
cyc_weights[:, 66:132] = 1

ped_weights = np.zeros((1, 204))
ped_weights[:, 132:] = 1

metric_weights = {
  'gate': gate_weights,
  'trn': trn_weights,
  'veh': veh_weights,
  'cyc': cyc_weights,
  'ped': ped_weights
}