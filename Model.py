import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
from matplotlib import cm
import math
from matplotlib.colors import LinearSegmentedColormap
# Import the RSS map processor
from find_in_map import RSSMapProcessor

class ImprovedRSSColorMapper:
    """Improved RSS color to dBm mapper using actual colormap lookup"""
    
    def __init__(self, min_dbm=-110.0, max_dbm=-40.0, colormap_name='jet'):
        self.min_dbm = min_dbm
        self.max_dbm = max_dbm
        self.colormap_name = colormap_name
        
        # Build lookup table for color to dBm mapping
        self._build_color_lookup()
        
        # Track actual data range for adaptive normalization
        self.actual_min_dbm = None
        self.actual_max_dbm = None
        
    def _build_color_lookup(self, n_colors=256):
        """Build a lookup table from colors to dBm values"""
        # Get the colormap
        cmap = cm.get_cmap(self.colormap_name)
        
        # Generate dBm values
        dbm_values = np.linspace(self.min_dbm, self.max_dbm, n_colors)
        
        # Generate corresponding colors (0-255 RGB)
        norm_values = np.linspace(0, 1, n_colors)
        colors_rgba = cmap(norm_values)
        self.lookup_colors = (colors_rgba[:, :3] * 255).astype(np.uint8)
        self.lookup_dbm = dbm_values
        
    def rgb_to_dbm_accurate(self, rgb_image):
        """Convert RGB to dBm using nearest neighbor in color space"""
        # Ensure input is uint8
        if rgb_image.max() > 1.0:
            rgb_uint8 = rgb_image.astype(np.uint8)
        else:
            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        
        # Reshape for distance calculation
        h, w = rgb_uint8.shape[:2]
        pixels = rgb_uint8.reshape(-1, 3)
        
        # Find nearest color in lookup table for each pixel
        distances = cdist(pixels, self.lookup_colors, metric='euclidean')
        nearest_indices = np.argmin(distances, axis=1)
        
        # Map to dBm values
        dbm_values = self.lookup_dbm[nearest_indices]
        dbm_image = dbm_values.reshape(h, w)
        
        # Update actual range
        self.actual_min_dbm = dbm_image.min()
        self.actual_max_dbm = dbm_image.max()
        
        return dbm_image
    
    def normalize_dbm_adaptive(self, dbm_values):
        """Normalize dBm values to [-1, 1] using actual data range"""
        if self.actual_min_dbm is not None and self.actual_max_dbm is not None:
            # Use actual data range for better normalization
            normalized = 2 * (dbm_values - self.actual_min_dbm) / (self.actual_max_dbm - self.actual_min_dbm) - 1
        else:
            # Fallback to full range
            normalized = 2 * (dbm_values - self.min_dbm) / (self.max_dbm - self.min_dbm) - 1
        
        return np.clip(normalized, -1, 1)
    
    def create_colorbar_reference(self):
        """Create a reference colorbar for visualization"""
        cmap = cm.get_cmap(self.colormap_name)
        
        # Create colorbar image
        n_levels = 256
        colorbar = np.linspace(0, 1, n_levels).reshape(-1, 1)
        colorbar = np.repeat(colorbar, 20, axis=1)
        
        # Apply colormap
        colorbar_rgb = cmap(colorbar)[:, :, :3]
        
        return colorbar_rgb, np.linspace(self.max_dbm, self.min_dbm, n_levels)


class RSSColorMapper:
    """Maps RSS colorbar colors to dBm values"""
    
    def __init__(self, min_dbm=-110.0, max_dbm=-40.0):
        self.min_dbm = min_dbm
        self.max_dbm = max_dbm
        
        # Define the colormap that matches your RSS map
        # This appears to be a jet-like colormap from blue (low) to red (high)
        colors = ['#0000FF', '#0080FF', '#00FFFF', '#00FF80', '#00FF00', 
                  '#80FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 100
        self.cmap = LinearSegmentedColormap.from_list('rss_map', colors, N=n_bins)
        
    def rgb_to_dbm(self, rgb_image):
        """Convert RGB RSS map to dBm values"""
        # Normalize RGB to [0,1]
        if rgb_image.max() > 1.0:
            rgb_norm = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_norm = rgb_image.astype(np.float32)
        
        # Convert to grayscale as a proxy for colormap position
        # This is a simplification - ideally we'd reverse-map from RGB to colormap
        gray = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2GRAY)
        
        # Map grayscale [0,1] to dBm range
        dbm_values = gray * (self.max_dbm - self.min_dbm) + self.min_dbm
        
        return dbm_values
    
    def normalize_dbm(self, dbm_values):
        """Normalize dBm values to [-1, 1] for neural network"""
        # Normalize to [0, 1]
        normalized = (dbm_values - self.min_dbm) / (self.max_dbm - self.min_dbm)
        # Scale to [-1, 1]
        normalized = 2 * normalized - 1
        return normalized

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerChannelDecoder(nn.Module):
    """Transformer decoder for processing channel features - UNCHANGED"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2):
        super(TransformerChannelDecoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Learnable queries for different aspects of the channel
        self.channel_queries = nn.Parameter(torch.randn(72, d_model))  # 72 spatial positions
        self.antenna_queries = nn.Parameter(torch.randn(4, d_model))   # 4 antennas
        self.frequency_queries = nn.Parameter(torch.randn(16, d_model)) # 16 frequency channels
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, memory, rss_features=None):
        """
        Args:
            memory: Encoded features from U-Net encoder (batch, seq_len, d_model)
            rss_features: RSS features (batch, d_model) - optional
        """
        batch_size = memory.shape[0]
        
        # Create queries by combining different types
        # Channel spatial queries
        spatial_queries = self.channel_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional encoding to memory
        memory = self.pos_encoder(memory.transpose(0, 1)).transpose(0, 1)
        
        # Add RSS information to queries if available
        if rss_features is not None:
            rss_expanded = rss_features.unsqueeze(1).expand(-1, spatial_queries.shape[1], -1)
            spatial_queries = spatial_queries + rss_expanded
        
        # Apply transformer decoder
        decoded = self.transformer_decoder(spatial_queries, memory)
        
        # Apply output projection and normalization
        decoded = self.output_proj(decoded)
        decoded = self.norm(decoded)
        
        return decoded

class EnhancedCrossAttention(nn.Module):
    """Enhanced cross-attention with better feature fusion and residual connection"""
    
    def __init__(self, channel_dim, rss_dim, hidden_dim=256, num_heads=8):
        super(EnhancedCrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention components
        self.channel_proj = nn.Linear(channel_dim, hidden_dim)
        self.rss_proj = nn.Linear(rss_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.out_proj = nn.Linear(hidden_dim, channel_dim)
        self.norm = nn.LayerNorm(channel_dim)
        
    def forward(self, channel_features, rss_features):
        # Project to common dimension
        channel_hidden = self.channel_proj(channel_features.unsqueeze(1))  # (batch, 1, hidden)
        rss_hidden = self.rss_proj(rss_features.unsqueeze(1))  # (batch, 1, hidden)
        
        # Cross-attention: channel attends to RSS
        attended, attention_weights = self.multihead_attn(
            query=channel_hidden,
            key=rss_hidden,
            value=rss_hidden
        )
        
        # Fuse original and attended features
        fused = torch.cat([channel_hidden, attended], dim=-1)
        fused = self.fusion(fused).squeeze(1)
        
        # Project back and add residual
        output = self.out_proj(fused)
        output = self.norm(channel_features + output)
        
        return output

class ResidualUNetBlock(nn.Module):
    """Enhanced U-Net block with residual connections"""
    
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, stride=2):
        super(ResidualUNetBlock, self).__init__()
        
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(8, out_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, stride, 1),
                nn.GroupNorm(8, out_channels),
                nn.LeakyReLU(0.2, True)
            )
            # Residual connection for downsampling
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.GroupNorm(8, out_channels)
            )
        else:
            if isinstance(stride, int):
                output_padding = 1 if stride == 2 else 0
            else:
                output_padding = tuple(1 if s == 2 else 0 for s in stride)
                
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, 
                                 output_padding=output_padding),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(True)
            )
            # Residual connection for upsampling
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, 0,
                                 output_padding=output_padding),
                nn.GroupNorm(8, out_channels)
            )
            
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = x + residual  # Add residual connection
        if self.use_dropout:
            x = self.dropout(x)
        return x

class ImprovedPhysicsInformedUNet(nn.Module):
    """Improved U-Net with transformer decoder, enhanced attention, and residual connections"""
    
    def __init__(self, channel_shape=(32, 4, 576), rss_size=64, latent_dim=256, use_dbm_values=True):
        super(ImprovedPhysicsInformedUNet, self).__init__()
        
        self.channel_shape = channel_shape
        self.use_dbm_values = use_dbm_values
        n_channels, n_antennas, n_taps = channel_shape
        
        # Enhanced encoder with residual connections
        self.enc1 = ResidualUNetBlock(32, 64, down=True, stride=2)
        self.enc2 = ResidualUNetBlock(64, 128, down=True, stride=2)
        self.enc3 = ResidualUNetBlock(128, 256, down=True, stride=(1, 2))
        
        # Additional encoder residual connections
        self.enc_residual1 = nn.Conv2d(32, 64, 1, 1, 0)
        self.enc_residual2 = nn.Conv2d(64, 128, 1, 1, 0)
        self.enc_residual3 = nn.Conv2d(128, 256, 1, 1, 0)
        
        # Enhanced RSS feature extractor with attention
        # Now expects 2 channels if use_dbm_values: normalized dBm values + original grayscale
        rss_input_channels = 2 if use_dbm_values else 1
        self.rss_encoder = nn.Sequential(
            nn.Conv2d(rss_input_channels, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced cross-attention
        self.cross_attention = EnhancedCrossAttention(
            channel_dim=256 * 72,
            rss_dim=256,
            hidden_dim=256,
            num_heads=8
        )
        
        # Reshape layer to prepare for transformer
        self.to_sequence = nn.Linear(256 * 72, 72 * 256)
        
        # Transformer decoder for better feature processing - UNCHANGED
        self.transformer_decoder = TransformerChannelDecoder(
            d_model=256,
            nhead=8,
            num_layers=5,
            dim_feedforward=1024,
            dropout=0.15
        )
        
        # Back to spatial representation
        self.from_sequence = nn.Linear(256, 256)
        
        # Enhanced decoder with residual connections
        self.dec1 = ResidualUNetBlock(256, 128, down=False, use_dropout=True, stride=(1, 2))
        self.dec2 = ResidualUNetBlock(256, 64, down=False, use_dropout=True, stride=2)  # 256 = 128 + 128 (skip)
        self.dec3 = ResidualUNetBlock(128, 32, down=False, stride=2)  # 128 = 64 + 64 (skip)
        
        # Skip connection processing
        self.skip_conv2 = nn.Conv2d(128, 128, 1)  # Match channels for skip connection
        self.skip_conv1 = nn.Conv2d(64, 64, 1)    # Match channels for skip connection
        
        # Final refinement with residual
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1)
        )
        
        # Global residual connection
        self.global_residual = nn.Conv2d(32, 32, 1)
        
    def forward(self, initial_channel, rss_map):
        batch_size = initial_channel.shape[0]
        
        # Store input for global residual
        input_residual = self.global_residual(initial_channel)
        
        # Encoder with skip connections storage
        e1 = self.enc1(initial_channel)  # (batch, 64, 2, 288)
        e2 = self.enc2(e1)               # (batch, 128, 1, 144)
        e3 = self.enc3(e2)               # (batch, 256, 1, 72)
        
        # Extract RSS features
        rss_features = self.rss_encoder(rss_map)
        rss_features = rss_features.view(batch_size, -1)
        
        # Flatten encoder output for cross-attention
        e3_flat = e3.view(batch_size, -1)
        
        # Enhanced cross-attention with residual
        attended = self.cross_attention(e3_flat, rss_features)
        
        # Prepare for transformer: reshape to sequence
        transformer_input = self.to_sequence(attended)
        transformer_input = transformer_input.view(batch_size, 72, 256)
        
        # Apply transformer decoder - UNCHANGED
        transformer_output = self.transformer_decoder(
            transformer_input, 
            rss_features
        )
        
        # Back to spatial representation
        processed = self.from_sequence(transformer_output)
        processed = processed.view(batch_size, 256, 1, 72)
        
        # Add residual from encoder output to transformer output
        processed = processed + e3
        
        # Enhanced decoder with skip connections
        d1 = self.dec1(processed)              # (batch, 128, 1, 144)
        e2_processed = self.skip_conv2(e2)     # Process skip connection
        d1_skip = torch.cat([d1, e2_processed], dim=1)  # (batch, 256, 1, 144)
        
        d2 = self.dec2(d1_skip)                # (batch, 64, 2, 288)
        e1_processed = self.skip_conv1(e1)     # Process skip connection
        d2_skip = torch.cat([d2, e1_processed], dim=1)  # (batch, 128, 2, 288)
        
        d3 = self.dec3(d2_skip)                # (batch, 32, 4, 576)
        
        # Final refinement
        output = self.final_conv(d3)
        
        # Add global residual connection
        output = output + input_residual
        
        return output

class PhysicsInformedLoss(nn.Module):
    """Combined loss function: NMSE + alpha * MSE(RSS, channel_power)"""
    
    def __init__(self, alpha=0.1, use_dbm_correlation=True):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha
        self.use_dbm_correlation = use_dbm_correlation
        
    def real_to_complex(self, real_tensor):
        """Convert real/imag tensor back to complex"""
        # Input shape: (batch, 32, 4, 576)
        # Split into real and imaginary parts
        n_channels = real_tensor.shape[1] // 2
        real_part = real_tensor[:, :n_channels, :, :]
        imag_part = real_tensor[:, n_channels:, :, :]
        
        # Combine into complex tensor
        complex_tensor = torch.complex(real_part, imag_part)
        return complex_tensor
        
    def calculate_nmse(self, pred, target):
        """Calculate Normalized Mean Square Error for complex channels"""
        # Convert to complex if needed
        if not pred.is_complex():
            pred = self.real_to_complex(pred)
            target = self.real_to_complex(target)
        
        # NMSE for complex values
        mse = torch.mean(torch.abs(pred - target) ** 2)
        target_power = torch.mean(torch.abs(target) ** 2)
        nmse = mse / (target_power)
        return nmse
    
    def calculate_channel_power(self, channel):
        """Calculate channel power across spatial dimensions"""
        # Convert to complex if needed
        if not channel.is_complex():
            channel = self.real_to_complex(channel)
        
        # Sum power across antenna and tap dimensions, keep frequency dimension
        power = torch.sum(torch.abs(channel) ** 2, dim=(2, 3))  # Shape: (batch, 16)
        return power
    
    def forward(self, pred_channel, true_channel, rss_map):
        # NMSE loss (handles complex conversion internally)
        nmse_loss = self.calculate_nmse(pred_channel, true_channel)
        
        if self.use_dbm_correlation and rss_map.shape[1] == 2:
            # Extract dBm values from the second channel of RSS map
            rss_dbm = rss_map[:, 1:2, :, :]  # Shape: (batch, 1, H, W)
            
            # Channel power in dB scale
            pred_power = self.calculate_channel_power(pred_channel)
            pred_power_db = 10 * torch.log10(pred_power)  # Convert to dB
            
            # Average RSS dBm values
            rss_avg_dbm = torch.mean(rss_dbm, dim=(2, 3))  # Shape: (batch, 1)
            
            # Normalize both for comparison
            pred_power_norm = F.normalize(pred_power_db, p=2, dim=1)
            rss_norm = F.normalize(rss_avg_dbm, p=2, dim=1)
            
            # MSE between normalized powers
            power_loss = F.mse_loss(pred_power_norm, rss_norm.expand_as(pred_power_norm))
        else:
            # Fallback to original method
            pred_power = self.calculate_channel_power(pred_channel)
            rss_avg = torch.mean(rss_map, dim=(2, 3))  # Shape: (batch, 1)
            
            pred_power_norm = F.normalize(pred_power, p=2, dim=1)
            rss_norm = F.normalize(rss_avg, p=2, dim=1)
            
            power_loss = F.mse_loss(pred_power_norm, rss_norm.expand_as(pred_power_norm))
        
        # Combined loss
        total_loss = nmse_loss + self.alpha * power_loss
        
        return total_loss, nmse_loss, power_loss

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                   best_val_nmse, checkpoint_path='checkpoint.pth'):
    """
    Save training checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_nmse': best_val_nmse
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path='checkpoint.pth'):
    """
    Load training checkpoint if it exists
    
    Returns:
        start_epoch, train_losses, val_losses, best_val_nmse
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_nmse = checkpoint['best_val_nmse']
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best validation NMSE so far: {best_val_nmse:.6f}")
        
        return start_epoch, train_losses, val_losses, best_val_nmse
    else:
        print("No checkpoint found, starting from scratch")
        return 0, [], [], float('inf')


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda',
                model_name_val = 'best_model.pth', model_name_train = 'last_model.pth', continue_=None):
    """Training loop"""
    save_frequency = 20
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.65)
    criterion = PhysicsInformedLoss(alpha=0.01)
    
    train_losses = []
    val_losses = []
    start_epoch = 0
    best_val_nmse = float('inf')
    if continue_:
        # Load checkpoint if exists
        start_epoch, train_losses, val_losses, best_val_nmse = load_checkpoint(
        model, optimizer, scheduler, model_name_train)
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_nmse = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for smomp, accurate, rss in pbar:
            smomp = smomp.to(device)
            accurate = accurate.to(device)
            rss = rss.to(device)
            #print(smomp.shape)
            #print(accurate.shape)
            optimizer.zero_grad()
            # Forward pass
            pred = model(smomp, rss)
            #print(pred)
            # Calculate loss
            loss, nmse, power_loss = criterion(pred, accurate, rss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            train_nmse += nmse.item()
            
            pbar.set_postfix({'Loss': loss.item(), 'NMSE': nmse.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_nmse = train_nmse / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_nmse = 0
        
        with torch.no_grad():
            for smomp, accurate, rss in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                smomp = smomp.to(device)
                accurate = accurate.to(device)
                rss = rss.to(device)
                
                pred = model(smomp, rss)
                loss, nmse, _ = criterion(pred, accurate, rss)
                
                val_loss += loss.item()
                val_nmse += nmse.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_nmse = val_nmse / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train NMSE={avg_train_nmse:.4f}, '
              f'Val Loss={avg_val_loss:.4f}, Val NMSE={avg_val_nmse:.4f}')

        # Save checkpoint periodically
        if (epoch + 1) % save_frequency == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, 
                          val_losses, best_val_nmse, model_name_train)
        # Save best model
        if avg_val_nmse < best_val_nmse:
            best_val_nmse = avg_val_nmse
            torch.save(model.state_dict(), model_name_val)
            print(f'Saved best model with Val NMSE={avg_val_nmse:.4f}')

    print('Last trained model saved')
    return train_losses, val_losses


def evaluate_test_set(model, test_loader, device='cuda'):
    """Evaluate model on test set with global normalization"""
    
    model = model.to(device)
    model.eval()
    
    total_nmse = 0
    n_samples = 0
    
    with torch.no_grad():
        for smomp, accurate, rss in tqdm(test_loader, desc='Testing'):
            smomp = smomp.to(device)
            accurate = accurate.to(device)
            rss = rss.to(device)
            
            pred = model(smomp, rss)
            
            # Convert to complex for NMSE calculation
            n_channels = pred.shape[1] // 2

            pred_complex = torch.complex(pred[:, :n_channels], pred[:, n_channels:])
            accurate_complex = torch.complex(accurate[:, :n_channels], accurate[:, n_channels:])
            # Calculate NMSE
            mse = torch.mean(torch.abs(pred_complex - accurate_complex) ** 2)
            # print(mse)
            target_power = torch.mean(torch.abs(accurate_complex) ** 2)
            nmse = mse / (target_power)
            total_nmse += nmse.item() * smomp.shape[0]
            n_samples += smomp.shape[0]
    
    avg_nmse = total_nmse / n_samples
    return avg_nmse


class GlobalNormalizedDataset(Dataset):
    """Dataset with global normalization applied before train/test split"""
    
    def __init__(self, smomp_file, accurate_file, user_positions_file, 
                 rss_processor, crop_size=30, split='train', 
                 train_ratio=0.7, val_ratio=0.15, random_seed=42, use_dbm_values=True):
        
        # Set random seed for reproducible splits
        np.random.seed(random_seed)
        
        # Load ALL data first
        self.smomp_channels = np.load(smomp_file)
        self.accurate_channels = np.load(accurate_file)
        
        # Convert complex to real and imaginary parts
        self.smomp_channels_real = np.concatenate([
            np.real(self.smomp_channels),
            np.imag(self.smomp_channels)
        ], axis=1)
        
        self.accurate_channels_real = np.concatenate([
            np.real(self.accurate_channels),
            np.imag(self.accurate_channels)
        ], axis=1)
        
        # GLOBAL NORMALIZATION - relative to max across ALL data
        self.smomp_max = np.max(np.abs(self.smomp_channels_real))
        self.accurate_max = np.max(np.abs(self.accurate_channels_real))
        # print(self.smomp_max)
        # print(self.accurate_max)
        # Normalize ALL data using global parameters
        self.smomp_channels_normalized = self.smomp_channels_real / max(self.smomp_max, self.accurate_max)
        self.accurate_channels_normalized = self.accurate_channels_real / max(self.smomp_max, self.accurate_max)
        
        # Load user positions
        self.user_positions = []
        with open(user_positions_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    x, y, z = map(float, line.strip().split())
                    self.user_positions.append((x, y))
        
        self.rss_processor = rss_processor
        self.crop_size = crop_size
        self.use_dbm_values = use_dbm_values
        
        # Initialize RSS color mapper
        self.rss_color_mapper = RSSColorMapper(min_dbm=-110.0, max_dbm=-40.0)
        
        # Create random splits AFTER normalization
        n_samples = len(self.smomp_channels_normalized)
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")
        
        print(f"Split '{split}': {len(self.indices)} samples")
        
        # Store normalization parameters for later use
        self.normalization_params = {
            'smomp_max': self.smomp_max,
            'accurate_max': self.accurate_max
        }
    
    def get_normalization_params(self):
        """Get normalization parameters for saving"""
        return self.normalization_params
    
    def denormalize_smomp(self, normalized_data):
        """Denormalize SMOMP data back to original scale"""
        return normalized_data * self.smomp_max
    
    def denormalize_accurate(self, normalized_data):
        """Denormalize accurate data back to original scale"""
        return normalized_data * self.accurate_max
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Get normalized channels (already normalized globally)
        smomp_channel = self.smomp_channels_normalized[real_idx]
        accurate_channel = self.accurate_channels_normalized[real_idx]
        
        # Get RSS map
        user_idx = real_idx % len(self.user_positions)
        user_x, user_y = self.user_positions[user_idx]
        rss_crop = self.rss_processor.crop_around_user(user_x, user_y, self.crop_size)
        
        if rss_crop is None:
            rss_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)
        
        if self.use_dbm_values:
            # Convert RGB to dBm values
            rss_dbm = self.rss_color_mapper.rgb_to_dbm(rss_crop)
            rss_dbm_normalized = self.rss_color_mapper.normalize_dbm(rss_dbm)
            
            # Also keep grayscale for texture information
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_gray_normalized = rss_gray.astype(np.float32) / 255.0
            
            # Stack both channels: grayscale and normalized dBm
            rss_tensor = torch.stack([
                torch.from_numpy(rss_gray_normalized).float(),
                torch.from_numpy(rss_dbm_normalized).float()
            ], dim=0)
        else:
            # Original grayscale only
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_normalized = rss_gray.astype(np.float32) / 255.0
            rss_tensor = torch.from_numpy(rss_normalized).unsqueeze(0).float()
        
        # Convert to tensors
        smomp_tensor = torch.from_numpy(smomp_channel).float()
        accurate_tensor = torch.from_numpy(accurate_channel).float()
        
        return smomp_tensor, accurate_tensor, rss_tensor


# Usage example:
def create_datasets(smomp_file, accurate_file, user_positions_file, rss_processor, use_dbm_values=True):
    """Create train, validation, and test datasets with consistent normalization"""
    
    # Create train dataset (this computes global normalization)
    train_dataset = GlobalNormalizedDataset(
        smomp_file, accurate_file, user_positions_file, rss_processor,
        split='train', train_ratio=0.8, val_ratio=0.1, use_dbm_values=use_dbm_values
    )
    
    # Get normalization params
    norm_params = train_dataset.get_normalization_params()
    print(f"Using normalization params: {norm_params}")
    
    # Create val and test datasets with same parameters
    # (they will use the same global normalization computed in train dataset)
    val_dataset = GlobalNormalizedDataset(
        smomp_file, accurate_file, user_positions_file, rss_processor,
        split='val', train_ratio=0.8, val_ratio=0.1, use_dbm_values=use_dbm_values
    )
    
    test_dataset = GlobalNormalizedDataset(
        smomp_file, accurate_file, user_positions_file, rss_processor,
        split='test', train_ratio=0.8, val_ratio=0.1, use_dbm_values=use_dbm_values
    )
    
    return train_dataset, val_dataset, test_dataset



