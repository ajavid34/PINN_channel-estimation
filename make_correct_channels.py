# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:15:32 2025

@author: sjavid

Build ground-truth per-tap MIMO channel tensors from a Wireless Insite
ray-tracing CSV. Run as a CLI; see `python make_correct_channels.py --help`.
"""

import argparse
import ast

import numpy as np
import pandas as pd

def safe_parse_list(value):
    try:
        return np.array(ast.literal_eval(value))  # Convert string to list safely
    except:
        return np.array([])  # Return empty array if parsing fails


# Raised cosine pulse function
def raised_cosine_pulse(t, Ts=1.0, beta=0.4):
    """
    Compute the raised cosine pulse response for a given delay.
    Handles cases where division by zero might occur.
    """
    # Avoid division by zero issues with isclose
    zero_indices = np.isclose(np.abs(2 * beta * t / Ts), 1)

    # Compute standard raised cosine
    numerator = np.sin(np.pi * t / Ts) * np.cos(beta * np.pi * t / Ts)
    denominator = (np.pi * t / Ts) * (1 - (2 * beta * t / Ts) ** 2)

    # Prevent division by zero
    pulse = np.zeros_like(t, dtype=float)
    non_zero_indices = ~np.isclose(denominator, 0)
    pulse[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]

    # For specific zero denominator case
    pulse[zero_indices] = np.sinc(1 / (2 * beta))

    # Additional safeguard: if t is exactly 0, use the analytical value
    t_zero_indices = np.isclose(t, 0)
    pulse[t_zero_indices] = 1.0

    return pulse


# Compute UPA array response for Tx and Rx
def array_response_UPA(theta_x, theta_y, N_x, N_y):
    """
    Computes 2D UPA response using Kronecker structure.
    """
    n_x = np.arange(N_x)
    n_y = np.arange(N_y)

    a_theta_x = np.exp(-1j * np.pi * n_x * np.cos(theta_y) * np.sin(theta_x))
    a_theta_y = np.exp(-1j * np.pi * n_y * np.sin(theta_y))

    return np.kron(a_theta_y, a_theta_x)  # (N_x * N_y,)


# Compute the complex gain per path
def make_complex_gain(path_gain, path_delay, path_phase, d, Bw=1e8):
    """
    Compute complex gain per path, using relative path delays to better model
    the temporal diversity of the channel.
    """
    # Create output array of zeros
    result = np.zeros_like(path_gain, dtype=complex)

    # Only calculate for non-zero path gains
    non_zero_mask = path_gain != 0

    if np.any(non_zero_mask):
        path_gain_nz = path_gain[non_zero_mask]
        path_delay_nz = path_delay[non_zero_mask]
        path_phase_nz = path_phase[non_zero_mask]

        # Calculate minimum delay to use as reference
        min_delay = np.min(path_delay_nz) if len(path_delay_nz) > 0 else 0

        # Use relative delays from the minimum delay path
        relative_delays = path_delay_nz - min_delay

        # Apply raised cosine filter with relative delays
        result[non_zero_mask] = path_gain_nz * np.exp(1j * path_phase_nz) * raised_cosine_pulse(
            d / Bw - relative_delays, 1 / Bw)

    return result


def build_channel_tensor(df, N_tx_x, N_tx_y, N_rx_x, N_rx_y, N_tap, Bw, Pt,
                          mask_low_gains=False, n_mask=15):
    """Run the ray-tracing-to-channel conversion for every row in ``df``."""
    channel_matrices = []

    for row_index in range(len(df)):
        # Extract channel parameters
        aod_phi = np.deg2rad(safe_parse_list(df.loc[row_index, "AOD_PHI"]))  # AoD Azimuth
        aod_theta = np.deg2rad(safe_parse_list(df.loc[row_index, "AOD_THETA"]))  # AoD Elevation
        aoa_phi = np.deg2rad(safe_parse_list(df.loc[row_index, "AOA_PHI"]))  # AoA Azimuth
        aoa_theta = np.deg2rad(safe_parse_list(df.loc[row_index, "AOA_THETA"]))  # AoA Elevation
        path_gain = safe_parse_list(df.loc[row_index, "Pathgain"]) + Pt  # Path Gain
        path_delay = safe_parse_list(df.loc[row_index, "ToA"])  # Path Delay
        path_phase = np.deg2rad(safe_parse_list(df.loc[row_index, "PHASE"]))  # Path Phase

        # Skip rows where parsing failed or arrays have different lengths
        if (len(aod_phi) == 0 or len(aod_theta) == 0 or
                len(aoa_phi) == 0 or len(aoa_theta) == 0 or
                len(path_gain) == 0 or len(path_delay) == 0 or
                len(path_phase) == 0):
            print(f"Skipping row {row_index} due to missing data")
            continue

        # Check if all arrays have the same length
        array_lens = [len(aod_phi), len(aod_theta), len(aoa_phi), len(aoa_theta),
                      len(path_gain), len(path_delay), len(path_phase)]
        if len(set(array_lens)) != 1:
            print(f"Skipping row {row_index} due to inconsistent array lengths: {array_lens}")
            continue

        if mask_low_gains:
            # Identify indices corresponding to the n lowest gains
            indices_lowest = np.argsort(path_gain)[:n_mask]
            # Create a boolean mask with True for paths to keep
            mask = np.ones(len(path_gain), dtype=bool)
            mask[indices_lowest] = False

            # Apply the mask to all related arrays
            aod_phi = aod_phi[mask]
            aod_theta = aod_theta[mask]
            aoa_phi = aoa_phi[mask]
            aoa_theta = aoa_theta[mask]
            path_gain = path_gain[mask]
            path_delay = path_delay[mask]
            path_phase = path_phase[mask]

        # Convert path gain from dB to linear scale
        # Important: Handle zero and negative dB values correctly
        path_gain_linear = np.zeros_like(path_gain, dtype=float)

        # Only convert positive dB values, leave zeros as zero
        positive_mask = path_gain > 0
        if np.any(positive_mask):
            path_gain_linear[positive_mask] = 10 ** (path_gain[positive_mask] / 10)

        # For negative dB values (attenuation), convert carefully
        negative_mask = path_gain < 0
        if np.any(negative_mask):
            path_gain_linear[negative_mask] = 10 ** (path_gain[negative_mask] / 10)

        # Zero dB values remain zero in linear scale
        # This is already handled by initializing path_gain_linear with zeros

        # Initialize channel matrix
        H = np.zeros((N_tap, N_rx_x * N_rx_y, N_tx_x * N_tx_y), dtype=complex)

        # Compute the channel response per tap
        for d in range(N_tap):
            cgain = make_complex_gain(path_gain_linear, path_delay, path_phase, d, Bw)

            # Compute the array response for each path
            for p in range(len(path_gain_linear)):
                # Skip paths with zero gain
                if path_gain_linear[p] == 0:
                    continue

                a_tx = array_response_UPA(aod_phi[p], aod_theta[p], N_tx_x, N_tx_y)  # (N_tx,)
                a_rx = array_response_UPA(aoa_phi[p], aoa_theta[p], N_rx_x, N_rx_y)  # (N_rx,)

                # Compute contribution of each path
                H[d, :, :] += cgain[p] * np.outer(a_rx, a_tx)

        # Store the computed channel matrix
        channel_matrices.append(H)

    # Shape: (num_snapshots, N_tap, N_rx_x*N_rx_y, N_tx_x*N_tx_y)
    return np.array(channel_matrices)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build ground-truth per-tap MIMO channel tensors from a "
                    "Wireless Insite ray-tracing CSV."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to the Wireless Insite concatenated CSV "
                             "(e.g. Dataset/15GHz_concatenated_data.csv).")
    parser.add_argument("--out", required=True,
                        help="Path where the resulting .npy channel tensor "
                             "will be saved (e.g. 3D_channel_15GHz_2x2_Pt50.npy).")
    parser.add_argument("--pt", type=float, default=50.0,
                        help="Transmit power Pt in dBm (default: 50). Must match "
                             "the RSS map used at training time.")
    parser.add_argument("--bw", type=float, default=4e8,
                        help="System bandwidth in Hz (default: 4e8 = 400 MHz, "
                             "the 15 GHz setting. Use 2e8 for the 8 GHz setting).")
    parser.add_argument("--n-tx-x", type=int, default=24,
                        help="Tx UPA size along x (default: 24).")
    parser.add_argument("--n-tx-y", type=int, default=24,
                        help="Tx UPA size along y (default: 24).")
    parser.add_argument("--n-rx-x", type=int, default=2,
                        help="Rx UPA size along x (default: 2).")
    parser.add_argument("--n-rx-y", type=int, default=2,
                        help="Rx UPA size along y (default: 2).")
    parser.add_argument("--n-tap", type=int, default=16,
                        help="Number of delay taps (default: 16).")
    parser.add_argument("--mask-low-gains", action="store_true",
                        help="If set, drop the N weakest paths per snapshot.")
    parser.add_argument("--n-mask", type=int, default=15,
                        help="Number of weakest paths to drop when "
                             "--mask-low-gains is set (default: 15).")
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"Loading ray-tracing CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    channel_matrices = build_channel_tensor(
        df,
        N_tx_x=args.n_tx_x, N_tx_y=args.n_tx_y,
        N_rx_x=args.n_rx_x, N_rx_y=args.n_rx_y,
        N_tap=args.n_tap, Bw=args.bw, Pt=args.pt,
        mask_low_gains=args.mask_low_gains, n_mask=args.n_mask,
    )

    # Check for NaN or Inf values
    if np.any(np.isnan(channel_matrices)) or np.any(np.isinf(channel_matrices)):
        print("Warning: NaN or Inf values found in channel matrices!")
        channel_matrices = np.nan_to_num(channel_matrices)

    # Print some statistics to help with debugging
    print("Channel matrix statistics:")
    print(f"  Shape: {channel_matrices.shape}")
    print(f"  Min value (real): {np.real(channel_matrices).min()}")
    print(f"  Max value (real): {np.real(channel_matrices).max()}")
    print(f"  Mean value (real): {np.real(channel_matrices).mean()}")
    print(f"  Std value (real): {np.real(channel_matrices).std()}")
    print(f"  Min value (imag): {np.imag(channel_matrices).min()}")
    print(f"  Max value (imag): {np.imag(channel_matrices).max()}")
    print(f"  Mean value (imag): {np.imag(channel_matrices).mean()}")
    print(f"  Std value (imag): {np.imag(channel_matrices).std()}")

    np.save(args.out, channel_matrices)
    print(f"Saved channel tensor to {args.out} "
          f"(shape={channel_matrices.shape}, Bw={args.bw}, Pt={args.pt} dBm)")


if __name__ == "__main__":
    main()