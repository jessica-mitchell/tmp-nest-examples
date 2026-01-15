# -*- coding: utf-8 -*-
#
# generate_figures_for_docs.py
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
Generate figures from pre-computed reference data
-------------------------------------------------

This script generates all figures for the ignore_and_fire example documentation
without requiring NEST to be installed. It reads pre-computed reference data
from the ``reference_data/`` directory.

This is intended for use in documentation builds (e.g., ReadTheDocs) where
NEST is not available.

To regenerate the reference data, run the full Snakemake workflow::

    snakemake -c4

Usage::

    python generate_figures_for_docs.py

The script will generate figures in the ``figures/`` directory.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, rcParams

# Directory paths
SCRIPT_DIR = Path(__file__).parent
REFERENCE_DATA_DIR = SCRIPT_DIR / "reference_data"
FIGURES_DIR = SCRIPT_DIR / "figures"


###############################################################################
# Data loading functions (NEST-independent)
###############################################################################


def load_parameters(neuron_model):
    """
    Load model parameters from JSON file.

    Parameters
    ----------
    neuron_model : str
        Name of the neuron model ('iaf_psc_alpha' or 'ignore_and_fire')

    Returns
    -------
    dict
        Parameter dictionary
    """
    param_file = REFERENCE_DATA_DIR / neuron_model / "parameters.json"
    with open(param_file) as f:
        return json.load(f)


def load_spike_data(neuron_model):
    """
    Load spike data from file.

    Parameters
    ----------
    neuron_model : str
        Name of the neuron model

    Returns
    -------
    numpy.ndarray
        Nx2 array with columns [sender_id, spike_time]
    """
    spike_file = REFERENCE_DATA_DIR / neuron_model / "spikes.dat"
    return np.loadtxt(spike_file, skiprows=3)


def load_connectivity_data(neuron_model, label):
    """
    Load connectivity data from file.

    Parameters
    ----------
    neuron_model : str
        Name of the neuron model
    label : str
        'presim' or 'postsim'

    Returns
    -------
    numpy.ndarray
        Nx4 array with columns [source, target, weight, delay]
    """
    conn_file = REFERENCE_DATA_DIR / neuron_model / f"connectivity_{label}.dat"
    return np.loadtxt(conn_file, skiprows=1)


def load_scaling_data():
    """
    Load pre-computed scaling experiment results.

    Returns
    -------
    dict
        Dictionary containing scaling data for both neuron models
    """
    scaling_file = REFERENCE_DATA_DIR / "scaling_results.json"
    with open(scaling_file) as f:
        return json.load(f)


###############################################################################
# Helper functions
###############################################################################


def get_connectivity_matrix(connectivity, pop_pre, pop_post):
    """
    Generate connectivity matrix from connectivity data.

    Parameters
    ----------
    connectivity : numpy.ndarray
        Nx4 array with columns [source, target, weight, delay]
    pop_pre : numpy.ndarray
        Array of presynaptic neuron IDs
    pop_post : numpy.ndarray
        Array of postsynaptic neuron IDs

    Returns
    -------
    numpy.ndarray
        Connectivity matrix of shape (len(pop_post), len(pop_pre))
    """
    W = np.zeros([len(pop_post), len(pop_pre)])

    for c in range(connectivity.shape[0]):
        source = connectivity[c, 0]
        target = connectivity[c, 1]
        weight = connectivity[c, 2]

        pre_idx = np.where(pop_pre == source)[0]
        post_idx = np.where(pop_post == target)[0]

        if len(pre_idx) > 0 and len(post_idx) > 0:
            W[post_idx[0], pre_idx[0]] = weight

    return W


def get_weight_distribution(connectivity, weight_bins):
    """
    Compute histogram of synaptic weights.

    Parameters
    ----------
    connectivity : numpy.ndarray
        Connectivity data array
    weight_bins : numpy.ndarray
        Bin edges for histogram

    Returns
    -------
    numpy.ndarray
        Normalized histogram values
    """
    return np.histogram(connectivity[:, 2], weight_bins, density=True)[0]


def time_and_population_averaged_spike_rate(spikes, time_interval, pop_size):
    """
    Calculate time and population averaged firing rate.

    Parameters
    ----------
    spikes : numpy.ndarray
        Spike data array
    time_interval : tuple
        (start_time, end_time) in ms
    pop_size : int
        Number of neurons

    Returns
    -------
    float
        Average firing rate in spikes/s
    """
    D = time_interval[1] - time_interval[0]
    n_events = np.sum(
        (spikes[:, 1] >= time_interval[0]) & (spikes[:, 1] <= time_interval[1])
    )
    return n_events / D * 1000.0 / pop_size


###############################################################################
# Plotting functions
###############################################################################


def plot_spikes(spikes, pars, output_path):
    """
    Create raster plot of spiking activity.
    """
    rate = time_and_population_averaged_spike_rate(
        spikes, (0.0, pars["T"]), pars["N_rec_spikes"]
    )

    plt.figure(num=1)
    plt.clf()
    plt.title(r"time and population averaged firing rate: $\nu=%.2f$ spikes/s" % rate)
    plt.plot(
        spikes[:, 1],
        spikes[:, 0],
        "o",
        ms=1,
        lw=0,
        mfc="k",
        mec="k",
        mew=0,
        alpha=0.2,
        rasterized=True,
    )

    plt.xlim(0, pars["T"])
    plt.ylim(0, pars["N_rec_spikes"])
    plt.xlabel(r"time (ms)")
    plt.ylabel(r"neuron id")

    plt.subplots_adjust(bottom=0.14, top=0.9, left=0.15, right=0.95)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")


def plot_connectivity_matrix(W, pop_pre, pop_post, pars, filename_label, output_path):
    """
    Plot connectivity matrix.
    """
    wmin = 0
    wmax = 150

    fig = plt.figure(num=2)
    plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[15, 1])

    matrix_ax = fig.add_subplot(gs[0])
    cmap = plt.cm.gray_r
    matrix = plt.pcolor(
        pop_pre, pop_post, W, cmap=cmap, rasterized=True, vmin=wmin, vmax=wmax
    )
    plt.xlabel(r"source id")
    plt.ylabel(r"target id")

    plt.xlim(pop_pre[0], pop_pre[-1])
    plt.ylim(pop_post[0], pop_post[-1])

    cb_ax = plt.subplot(gs[1])
    cb = plt.colorbar(matrix, cax=cb_ax, extend="max")
    cb.set_label(r"synaptic weight (pA)")

    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.1)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")


def plot_weight_distributions(whist_presim, whist_postsim, weights, pars, output_path):
    """
    Plot distributions of synaptic weights before and after simulation.
    """
    fig = plt.figure(num=3)
    plt.clf()
    lw = 3
    clr = ["0.6", "0.0"]
    plt.plot(weights[:-1], whist_presim, lw=lw, color=clr[0], label=r"pre sim.")
    plt.plot(weights[:-1], whist_postsim, lw=lw, color=clr[1], label=r"post sim.")
    plt.setp(plt.gca(), yscale="log")
    plt.legend(loc=1)
    plt.xlabel(r"synaptic weight (pA)")
    plt.ylabel(r"rel. frequency")
    plt.xlim(weights[0], weights[-2])
    plt.ylim(5e-5, 3)
    plt.subplots_adjust(bottom=0.14, top=0.9, left=0.15, right=0.95)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")


def plot_scaling_results(scaling_data, output_path):
    """
    Plot scaling experiment results.
    """
    neuron_models = scaling_data["neuron_models"]
    Ns = np.array(scaling_data["network_sizes"])

    sim_time = np.array(scaling_data["sim_time"])
    rate = np.array(scaling_data["rate"])
    weight_mean = np.array(scaling_data["weight_mean"])
    weight_sd = np.array(scaling_data["weight_sd"])

    fig = plt.figure(num=4)
    plt.clf()

    gs = gridspec.GridSpec(3, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ms = 4
    lw = 2
    clrs = ["0", "0.8"]

    for cm, neuron_model in enumerate(neuron_models):
        # Simulation time
        ax1.plot(
            Ns,
            sim_time[cm, :],
            "-o",
            mfc=clrs[cm],
            mec=clrs[cm],
            ms=ms,
            lw=lw,
            color=clrs[cm],
            label=r"\texttt{%s}" % neuron_model,
        )

        # Firing rate
        ax2.plot(
            Ns,
            rate[cm, :],
            "-o",
            mfc=clrs[cm],
            mec=clrs[cm],
            ms=ms,
            lw=lw,
            color=clrs[cm],
        )

        # Weight statistics
        if cm == 0:
            lbl1 = "mean"
            lbl2 = "mean + SD"
        else:
            lbl1 = ""
            lbl2 = ""

        ax3.plot(
            Ns,
            weight_mean[cm, :],
            "-o",
            mfc=clrs[cm],
            mec=clrs[cm],
            lw=lw,
            color=clrs[cm],
            ms=ms,
            label=lbl1,
        )
        ax3.plot(
            Ns,
            weight_mean[cm, :] + weight_sd[cm, :],
            "--",
            mfc=clrs[cm],
            mec=clrs[cm],
            lw=lw,
            color=clrs[cm],
            ms=ms,
            label=lbl2,
        )

    ax1.set_xlim(Ns[0], Ns[-1])
    ax1.set_xticklabels([])
    ax1.set_ylabel(r"simulation time (s)")
    ax1.legend(loc=2)

    ax2.set_xlim(Ns[0], Ns[-1])
    ax2.set_xticklabels([])
    ax2.set_ylim(0.5, 2)
    ax2.set_ylabel(r"firing rate (1/s)")

    ax3.set_xlim(Ns[0], Ns[-1])
    ax3.set_ylim(10, 100)
    ax3.set_xlabel(r"network size $N$")
    ax3.set_ylabel(r"synaptic weight (pA)")
    ax3.legend(loc=1)

    plt.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.95, hspace=0.1)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")


###############################################################################
# Main figure generation
###############################################################################


def generate_reference_figures(neuron_model):
    """
    Generate all reference figures for a given neuron model.

    Parameters
    ----------
    neuron_model : str
        'iaf_psc_alpha' or 'ignore_and_fire'
    """
    print(f"\nGenerating figures for {neuron_model}...")

    # Load data
    pars = load_parameters(neuron_model)
    spikes = load_spike_data(neuron_model)
    connectivity_presim = load_connectivity_data(neuron_model, "presim")
    connectivity_postsim = load_connectivity_data(neuron_model, "postsim")

    # Generate spike raster plot
    plot_spikes(
        spikes,
        pars,
        FIGURES_DIR / f"TwoPopulationNetworkPlastic_{neuron_model}_spikes.png",
    )

    # Generate connectivity matrices
    # Use first 100 excitatory neurons for visualization
    subset_size = 100
    # Node IDs start at 1 in NEST
    pop_E_start = 1
    pop_pre = np.arange(pop_E_start, pop_E_start + subset_size)
    pop_post = np.arange(pop_E_start, pop_E_start + subset_size)

    W_presim = get_connectivity_matrix(connectivity_presim, pop_pre, pop_post)
    W_postsim = get_connectivity_matrix(connectivity_postsim, pop_pre, pop_post)

    plot_connectivity_matrix(
        W_presim,
        pop_pre,
        pop_post,
        pars,
        "_presim",
        FIGURES_DIR / f"TwoPopulationNetworkPlastic_{neuron_model}_connectivity_presim.png",
    )
    plot_connectivity_matrix(
        W_postsim,
        pop_pre,
        pop_post,
        pars,
        "_postsim",
        FIGURES_DIR / f"TwoPopulationNetworkPlastic_{neuron_model}_connectivity_postsim.png",
    )

    # Generate weight distribution plot
    weights = np.arange(0.0, 150.1, 0.5)
    whist_presim = get_weight_distribution(connectivity_presim, weights)
    whist_postsim = get_weight_distribution(connectivity_postsim, weights)

    plot_weight_distributions(
        whist_presim,
        whist_postsim,
        weights,
        pars,
        FIGURES_DIR / f"TwoPopulationNetworkPlastic_{neuron_model}_weight_distributions.png",
    )


def generate_scaling_figure():
    """
    Generate scaling experiment figure from pre-computed results.
    """
    print("\nGenerating scaling figure...")
    scaling_data = load_scaling_data()
    plot_scaling_results(scaling_data, FIGURES_DIR / "scaling.png")


def main():
    """
    Main entry point for figure generation.
    """
    # Configure matplotlib
    rcParams["figure.figsize"] = (4, 3)
    rcParams["figure.dpi"] = 300
    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["axes.titlesize"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["xtick.labelsize"] = 8
    rcParams["text.usetex"] = True

    # Create figures directory
    FIGURES_DIR.mkdir(exist_ok=True)

    # Check for reference data
    if not REFERENCE_DATA_DIR.exists():
        print(f"ERROR: Reference data directory not found: {REFERENCE_DATA_DIR}")
        print("\nTo generate reference data, run the full Snakemake workflow:")
        print("    snakemake -c4")
        return 1

    # Generate figures for both neuron models
    for neuron_model in ["iaf_psc_alpha", "ignore_and_fire"]:
        model_data_dir = REFERENCE_DATA_DIR / neuron_model
        if model_data_dir.exists():
            generate_reference_figures(neuron_model)
        else:
            print(f"WARNING: No reference data for {neuron_model}, skipping...")

    # Generate scaling figure
    scaling_file = REFERENCE_DATA_DIR / "scaling_results.json"
    if scaling_file.exists():
        generate_scaling_figure()
    else:
        print("WARNING: No scaling data found, skipping scaling figure...")

    print("\nFigure generation complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
