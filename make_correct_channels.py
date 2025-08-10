# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:15:32 2025

@author: sjavid
"""

import numpy as np
import pandas as pd
import ast

# Load dataset
file_path = "concatenated_xx.csv"
df = pd.read_csv(file_path)

# System parameters
N_tx_x, N_tx_y = 24, 24  # Tx UPA (32x32 elements)
N_rx_x, N_rx_y = 2, 2  # Rx ULA (2x2 elements)
N_tap = 16  # Number of delay taps
Bw = 4e8  # Bandwidth
# Initialize lists to store processed results
channel_matrices = []
MASK = False
n_mas = 15
Pt = 50

# Function to safely parse lists from the dataframe
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


# Process each row (channel snapshot)
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

    if MASK:
        # Identify indices corresponding to the n lowest gains
        indices_lowest = np.argsort(path_gain)[:n_mas]
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

            a_tx = array_response_UPA(aod_phi[p], aod_theta[p], N_tx_x, N_tx_y)  # (1024,)
            a_rx = array_response_UPA(aoa_phi[p], aoa_theta[p], N_rx_x, N_rx_y)  # (4,)

            # Compute contribution of each path
            H[d, :, :] += cgain[p] * np.outer(a_rx, a_tx)  # Shape (4, 1024)

    # Store the computed channel matrix
    channel_matrices.append(H)

# Convert list to numpy array
channel_matrices = np.array(channel_matrices)  # Shape should be (num_snapshots, N_tap, N_rx_x*N_rx_y, N_tx_x*N_tx_y)

# Check for NaN or Inf values
if np.any(np.isnan(channel_matrices)) or np.any(np.isinf(channel_matrices)):
    print("Warning: NaN or Inf values found in channel matrices!")
    # Replace NaN/Inf with zeros
    channel_matrices = np.nan_to_num(channel_matrices)

# Print some statistics to help with debugging
print(f"Channel matrix statistics:")
print(f"  Shape: {channel_matrices.shape}")
print(f"  Min value (real): {np.real(channel_matrices).min()}")
print(f"  Max value (real): {np.real(channel_matrices).max()}")
print(f"  Mean value (real): {np.real(channel_matrices).mean()}")
print(f"  Std value (real): {np.real(channel_matrices).std()}")
print(f"  Min value (imag): {np.imag(channel_matrices).min()}")
print(f"  Max value (imag): {np.imag(channel_matrices).max()}")
print(f"  Mean value (imag): {np.imag(channel_matrices).mean()}")
print(f"  Std value (imag): {np.imag(channel_matrices).std()}")

# Save the channel matrices
np.save("3D_channel_15GHz_2x2_Pt50.npy", channel_matrices)

print(f"Saved channel model with shape: {channel_matrices.shape} and Bandwidth:{Bw}")