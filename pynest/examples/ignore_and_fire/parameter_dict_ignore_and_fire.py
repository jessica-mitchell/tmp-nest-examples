# -*- coding: utf-8 -*-
#
# parameter_dict-ignore_and_fire.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
"""
Parameter dictionary
--------------------

Default parameters for balanced random network with STDP synapses (TwoPopulationNetworkPlastic)

"""


pars = {}

pars["model_name"] = "TwoPopulationNetworkPlastic"  # Network model name

# network and connectivity parameters
pars["N"] = 12500  # total number of neurons
pars["K"] = 1250  # total number of inputs per neuron from local network
pars["beta"] = 0.8  # fraction of excitatory neurons/inputs

pars["allow_autapses"] = False
pars["allow_multapses"] = True

# neuron parameters

pars["neuron_model"] = "ignore_and_fire"

pars["E_L"] = 0.0
pars["C_m"] = 250.0
pars["tau_m"] = 20.0
pars["t_ref"] = 2.0
pars["theta"] = 20.0
pars["V_reset"] = 0.0

pars["ignore_and_fire_pars"] = {}
pars["ignore_and_fire_pars"]["rate_dist"] = [0.5, 1.5]
pars["ignore_and_fire_pars"]["phase_dist"] = [0.01, 1.0]

pars["I_DC"] = 0.0
pars["eta"] = 1.2

# synapse parameters
pars["J_E"] = 0.5  # EPSP amplitude (mV)
pars["g"] = 10.0  # relative IPSP amplitude (JI=-g*JE)
pars["delay"] = 1.5  # spike transmission delay (ms)
pars["tau_s"] = 2.0  # synaptic time constant (ms)

pars["stdp_alpha"] = 0.1  # relative magnitude of weight update for acausal firing
pars["stdp_lambda"] = 20.0  # magnitude of weight update for causal firing
pars["stdp_mu_plus"] = 0.4  # weight dependence exponent for causal firing
pars["stdp_tau_plus"] = 15.0  # time constant of weight update for causal firing (ms)
pars["stdp_tau_minus"] = 30.0  # time constant of weight update for acausal firing (ms)
pars["stdp_w_0"] = 1.0  # reference weight (pA)

# initial conditions
pars["V_init_min"] = pars["E_L"]  # min of initial membrane potential (mV)
pars["V_init_max"] = pars["theta"]  # max of initial membrane potential (mV)

# data recording
pars["record_spikes"] = False
pars["N_rec_spikes"] = "all"

pars["T"] = 10000.0
pars["dt"] = 2**-3
pars["tics_per_step"] = 2**7
pars["seed"] = 1
pars["n_threads"] = 4
pars["print_simulation_progress"] = True

pars["data_path"] = "data"
