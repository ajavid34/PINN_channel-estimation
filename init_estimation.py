import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm

class LSOFDMChannelEstimator:
    """
    LS OFDM Channel Estimation
    
    This implements the standard LS estimation in OFDM systems where:
    - Pilots are inserted at specific subcarriers in frequency domain
    - LS estimation is performed at pilot positions
    - Interpolation is used to estimate channel at data subcarriers
    """
    
    def __init__(self, N_tap=16, N_rx=4, N_tx=576, N_subcarriers=1024, 
                 pilot_spacing=4, SNR_dB=10):
        """
        Initialize LS OFDM channel estimator.
        
        Args:
            N_tap: Number of delay taps (time domain channel length)
            N_rx: Number of receive antennas 
            N_tx: Number of transmit antennas
            N_subcarriers: Number of OFDM subcarriers (should be >= N_tx)
            pilot_spacing: Pilot subcarrier spacing (e.g., 4 means pilot every 4th subcarrier)
            SNR_dB: Signal-to-noise ratio in dB
        """
        self.N_tap = N_tap
        self.N_rx = N_rx
        self.N_tx = N_tx
        self.N_subcarriers = N_subcarriers
        self.pilot_spacing = pilot_spacing
        self.SNR_dB = SNR_dB
        
        # Pilot positions in frequency domain
        self.pilot_positions = np.arange(0, N_subcarriers, pilot_spacing)
        self.N_pilots = len(self.pilot_positions)
        
        # Data positions
        self.data_positions = np.setdiff1d(np.arange(N_subcarriers), self.pilot_positions)
        
        print(f"LS OFDM Estimator initialized:")
        print(f"  Subcarriers: {N_subcarriers}")
        print(f"  Pilot subcarriers: {self.N_pilots} (spacing: {pilot_spacing})")
        print(f"  Data subcarriers: {len(self.data_positions)}")
        
    def time_to_frequency(self, h_time):
        """
        Convert time domain channel to frequency domain.
        
        Args:
            h_time: Time domain channel (N_tap, N_rx, N_tx)
            
        Returns:
            H_freq: Frequency domain channel (N_subcarriers, N_rx, N_tx)
        """
        # Zero-pad to N_subcarriers length
        h_padded = np.zeros((self.N_subcarriers, self.N_rx, self.N_tx), dtype=complex)
        h_padded[:self.N_tap, :, :] = h_time
        
        # FFT along first dimension (time/tap dimension)
        H_freq = np.fft.fft(h_padded, axis=0)
        
        return H_freq
    
    def frequency_to_time(self, H_freq):
        """
        Convert frequency domain channel to time domain.
        
        Args:
            H_freq: Frequency domain channel (N_subcarriers, N_rx, N_tx)
            
        Returns:
            h_time: Time domain channel (N_tap, N_rx, N_tx)
        """
        # IFFT
        h_time_full = np.fft.ifft(H_freq, axis=0)
        
        # Keep only first N_tap samples
        h_time = h_time_full[:self.N_tap, :, :]
        
        return h_time
    
    def estimate_channel(self, true_channel):
        """
        Perform LS OFDM channel estimation.
        
        Args:
            true_channel: True time-domain channel (N_tap, N_rx, N_tx)
            
        Returns:
            estimated_channel: Estimated time-domain channel (N_tap, N_rx, N_tx)
        """
        # Convert true channel to frequency domain
        H_true_freq = self.time_to_frequency(true_channel)
        
        # Signal and noise power
        signal_power = np.mean(np.abs(H_true_freq)**2)
        noise_power = signal_power / (10**(self.SNR_dB/10))
        noise_std = np.sqrt(noise_power/2)
        
        # Initialize estimated frequency domain channel
        H_est_freq = np.zeros((self.N_subcarriers, self.N_rx, self.N_tx), dtype=complex)
        
        # For each Rx-Tx pair
        for rx in range(self.N_rx):
            for tx in range(self.N_tx):
                # True frequency response
                H_true = H_true_freq[:, rx, tx]
                
                # Extract pilot subcarriers
                H_pilots = H_true[self.pilot_positions]
                
                # Add noise to pilot measurements
                noise = noise_std * (np.random.randn(self.N_pilots) + 
                                   1j * np.random.randn(self.N_pilots))
                H_pilots_noisy = H_pilots + noise
                
                # LS estimation at pilot positions (simple: H_est = Y/X for known pilot X=1)
                # In practice, pilots have known values, here we assume unit pilots
                H_ls_pilots = H_pilots_noisy  # Since we assume X=1
                
                # Interpolation to all subcarriers
                # Using different interpolation methods for magnitude and phase
                
                # Method 1: Linear interpolation (simple but effective)
                if self.N_pilots > 1:
                    # Interpolate magnitude and phase separately
                    mag_interp = interp1d(self.pilot_positions, np.abs(H_ls_pilots), 
                                         kind='linear', fill_value='extrapolate')
                    phase_interp = interp1d(self.pilot_positions, np.unwrap(np.angle(H_ls_pilots)), 
                                           kind='linear', fill_value='extrapolate')
                    
                    # Estimate at all subcarriers
                    all_positions = np.arange(self.N_subcarriers)
                    mag_est = mag_interp(all_positions)
                    phase_est = phase_interp(all_positions)
                    
                    # Combine magnitude and phase
                    H_est_freq[:, rx, tx] = mag_est * np.exp(1j * phase_est)
                else:
                    # If only one pilot, use constant channel
                    H_est_freq[:, rx, tx] = H_ls_pilots[0]
        
        # Convert back to time domain
        estimated_channel = self.frequency_to_time(H_est_freq)
        
        return estimated_channel
    
    def estimate_channel_with_smoothing(self, true_channel):
        """
        LS OFDM estimation with frequency domain smoothing.
        This exploits the fact that wireless channels are often smooth in frequency.
        """
        # Convert true channel to frequency domain
        H_true_freq = self.time_to_frequency(true_channel)
        
        # Signal and noise power
        signal_power = np.mean(np.abs(H_true_freq)**2)
        noise_power = signal_power / (10**(self.SNR_dB/10))
        noise_std = np.sqrt(noise_power/2)
        
        # Initialize
        H_est_freq = np.zeros((self.N_subcarriers, self.N_rx, self.N_tx), dtype=complex)
        
        for rx in range(self.N_rx):
            for tx in range(self.N_tx):
                # Get true frequency response
                H_true = H_true_freq[:, rx, tx]
                
                # Pilot measurements with noise
                H_pilots = H_true[self.pilot_positions]
                noise = noise_std * (np.random.randn(self.N_pilots) + 
                                   1j * np.random.randn(self.N_pilots))
                H_pilots_noisy = H_pilots + noise
                
                # Method 2: DFT-based interpolation (exploits time-domain sparsity)
                # Place pilots in full frequency grid
                H_pilot_grid = np.zeros(self.N_subcarriers, dtype=complex)
                H_pilot_grid[self.pilot_positions] = H_pilots_noisy
                
                # Transform to time domain
                h_time_est = np.fft.ifft(H_pilot_grid)
                
                # Keep only significant taps (denoising)
                # Threshold based on noise level
                threshold = 2 * noise_std / np.sqrt(self.N_pilots)
                h_time_est[np.abs(h_time_est) < threshold] = 0
                
                # Also enforce known channel length
                h_time_est[self.N_tap:] = 0
                
                # Transform back to frequency
                H_est_freq[:, rx, tx] = np.fft.fft(h_time_est)
        
        # Convert to time domain
        estimated_channel = self.frequency_to_time(H_est_freq)
        
        return estimated_channel
    
    def estimate_channel_mmse(self, true_channel, channel_correlation=None):
        """
        MMSE OFDM channel estimation (better than LS but requires channel statistics).
        
        Args:
            true_channel: True channel
            channel_correlation: Channel correlation matrix (if known)
        """
        # Convert to frequency domain
        H_true_freq = self.time_to_frequency(true_channel)
        
        # Signal and noise power
        signal_power = np.mean(np.abs(H_true_freq)**2)
        noise_power = signal_power / (10**(self.SNR_dB/10))
        
        # If no correlation provided, assume exponential correlation
        if channel_correlation is None:
            # Simple exponential correlation model in frequency
            rho = 0.95
            R_HH = np.zeros((self.N_subcarriers, self.N_subcarriers), dtype=complex)
            for i in range(self.N_subcarriers):
                for j in range(self.N_subcarriers):
                    R_HH[i, j] = signal_power * (rho ** abs(i - j))
        else:
            R_HH = channel_correlation
        
        # Extract pilot rows/columns from correlation matrix
        R_HP = R_HH[:, self.pilot_positions]  # All subcarriers to pilots
        R_PP = R_HH[np.ix_(self.pilot_positions, self.pilot_positions)]  # Pilot to pilot
        
        # MMSE interpolation matrix
        W_MMSE = R_HP @ np.linalg.inv(R_PP + (noise_power / signal_power) * np.eye(self.N_pilots))
        
        # Estimate channel
        H_est_freq = np.zeros((self.N_subcarriers, self.N_rx, self.N_tx), dtype=complex)
        
        for rx in range(self.N_rx):
            for tx in range(self.N_tx):
                # Get pilot measurements
                H_true = H_true_freq[:, rx, tx]
                H_pilots = H_true[self.pilot_positions]
                
                # Add noise
                noise = np.sqrt(noise_power/2) * (np.random.randn(self.N_pilots) + 
                                                  1j * np.random.randn(self.N_pilots))
                H_pilots_noisy = H_pilots + noise
                
                # MMSE estimation
                H_est_freq[:, rx, tx] = W_MMSE @ H_pilots_noisy
        
        # Convert to time domain
        estimated_channel = self.frequency_to_time(H_est_freq)
        
        return estimated_channel


def create_ls_ofdm_estimates(true_channels_file, output_file, 
                            N_subcarriers=1024, pilot_spacing=4, 
                            SNR_dB=10, method='smoothing'):
    """
    Create LS OFDM channel estimates.
    
    Args:
        true_channels_file: Path to true channel .npy file
        output_file: Path to save estimated channels
        N_subcarriers: Number of OFDM subcarriers
        pilot_spacing: Pilot spacing (e.g., 4 = pilot every 4th subcarrier)
        SNR_dB: Signal-to-noise ratio
        method: 'basic', 'smoothing', or 'mmse'
    """
    # Load true channels
    true_channels = np.load(true_channels_file)
    N_samples, N_tap, N_rx, N_tx = true_channels.shape
    
    print(f"Loaded channels: {true_channels.shape}")
    print(f"Using LS OFDM estimation with {method} method")
    
    # Initialize estimator
    estimator = LSOFDMChannelEstimator(
        N_tap=N_tap,
        N_rx=N_rx,
        N_tx=N_tx,
        N_subcarriers=N_subcarriers,
        pilot_spacing=pilot_spacing,
        SNR_dB=SNR_dB
    )
    
    # Estimate channels
    estimated_channels = np.zeros_like(true_channels)
    
    for i in tqdm(range(N_samples), desc="Estimating channels"):
        if method == 'basic':
            estimated_channels[i] = estimator.estimate_channel(true_channels[i])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Calculate NMSE
    mse = np.mean(np.abs(estimated_channels - true_channels)**2)
    true_power = np.mean(np.abs(true_channels)**2)
    nmse = mse / true_power
    nmse_db = 10 * np.log10(nmse)
    
    print(f"\nEstimation quality:")
    print(f"  NMSE: {nmse:.6f} ({nmse_db:.2f} dB)")
    
    # Save
    np.save(output_file, estimated_channels)
    print(f"Saved to: {output_file}")
    
    return estimated_channels


class InitialChannelEstimator:
    """
    Initial channel estimation algorithm to replace SMOMP.
    Uses pilot-based estimation with realistic impairments.
    """
    
    def __init__(self, N_tap=16, N_rx=4, N_tx=576, N_pilots=64, SNR_dB=10):
        """
        Initialize the channel estimator.
        
        Args:
            N_tap: Number of delay taps (16)
            N_rx: Number of receive antennas (4)
            N_tx: Number of transmit antennas (576)
            N_pilots: Number of pilot symbols per antenna
            SNR_dB: Signal-to-noise ratio in dB for pilot transmission
        """
        self.N_tap = N_tap
        self.N_rx = N_rx
        self.N_tx = N_tx
        self.N_pilots = N_pilots
        self.SNR_dB = SNR_dB
        
        # Generate orthogonal pilot sequences using DFT matrix
        self.generate_pilot_patterns()
        
    def generate_pilot_patterns(self):
        """Generate orthogonal pilot patterns for channel estimation."""
        # For large antenna arrays, we use a subset of antennas for pilots
        # This simulates practical limitations in pilot overhead
        
        # Select pilot positions (uniformly spaced)
        self.pilot_spacing = max(1, self.N_tx // self.N_pilots)
        self.pilot_positions = np.arange(0, self.N_tx, self.pilot_spacing)[:self.N_pilots]
        
        # Generate orthogonal pilot sequences (using DFT matrix)
        # Each row is a pilot sequence for one time slot
        if self.N_pilots <= 64:
            # For small pilot sets, use full orthogonal sequences
            dft_matrix = dft(self.N_pilots, scale='sqrtn')
            self.pilot_matrix = dft_matrix[:self.N_tap, :]  # Use N_tap rows
        else:
            # For large pilot sets, use random orthogonal sequences
            self.pilot_matrix = np.random.randn(self.N_tap, self.N_pilots) + \
                               1j * np.random.randn(self.N_tap, self.N_pilots)
            # Normalize each sequence
            for i in range(self.N_tap):
                self.pilot_matrix[i] /= np.linalg.norm(self.pilot_matrix[i])
    
    def estimate_channel(self, true_channel, method='ls_with_interpolation'):
        """
        Estimate the channel from the true channel with realistic impairments.
        
        Args:
            true_channel: True channel matrix (N_tap, N_rx, N_tx) complex
            method: Estimation method ('ls_with_interpolation', 'compressed_sensing', 'noisy')
            
        Returns:
            estimated_channel: Estimated channel matrix (N_tap, N_rx, N_tx) complex
        """
        if method == 'ls_with_interpolation':
            return self.ls_estimation_with_interpolation(true_channel)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def ls_estimation_with_interpolation(self, true_channel):
        """
        Least Squares estimation with limited pilots and interpolation.
        This simulates realistic pilot-based channel estimation.
        """
        estimated_channel = np.zeros_like(true_channel)
        
        # Signal power from true channel
        signal_power = np.mean(np.abs(true_channel)**2)
        noise_power = signal_power / (10**(self.SNR_dB/10))
        
        for tap in range(self.N_tap):
            for rx in range(self.N_rx):
                # Extract true channel for this tap and antenna
                h_true = true_channel[tap, rx, :]
                
                # Simulate pilot transmission at selected positions
                h_pilots = h_true[self.pilot_positions]
                
                # Add noise to pilot measurements
                noise = np.sqrt(noise_power/2) * (np.random.randn(self.N_pilots) + 
                                                  1j * np.random.randn(self.N_pilots))
                h_pilots_noisy = h_pilots + noise
                
                # Interpolate to all antenna positions
                # Using linear interpolation in magnitude and phase
                magnitude = np.abs(h_pilots_noisy)
                phase = np.angle(h_pilots_noisy)
                
                # Interpolate magnitude and phase separately
                all_positions = np.arange(self.N_tx)
                interp_magnitude = np.interp(all_positions, self.pilot_positions, magnitude)
                
                # Unwrap phase for better interpolation
                phase_unwrapped = np.unwrap(phase)
                interp_phase = np.interp(all_positions, self.pilot_positions, phase_unwrapped)
                
                # Reconstruct complex channel
                estimated_channel[tap, rx, :] = interp_magnitude * np.exp(1j * interp_phase)
                
                # # Add estimation error due to interpolation
                # interp_error = np.sqrt(noise_power) * (np.random.randn(self.N_tx) + 
                #                                        1j * np.random.randn(self.N_tx))
                # estimated_channel[tap, rx, :] += 0.5 * interp_error
        
        return estimated_channel