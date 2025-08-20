#!/usr/bin/env python3
# ==============================================================================
# === Enhanced 4D Radar Processing Dashboard with Advanced Drone Detection ===
# ==============================================================================
# 
# Copyright (C) 2025 Joshua Onuegbu
# Enhanced radar processing system with integration of Jon Kraft's CN0566 algorithms
#
# This implementation incorporates and builds upon the excellent work by Jon Kraft
# from Analog Devices Inc., specifically drawing from:
#   - FMCW_RADAR_Waterfall.py (FMCW processing fundamentals)
#   - FMCW_RADAR_Waterfall_ChirpSync.py (TDD synchronization)
#   - FMCW_RADAR_Waterfall_RangeDisplay.py (Range processing)
#   - FMCW_Velocity_RADAR_Waterfall.py (Velocity analysis)
#   - CFAR_RADAR_Waterfall.py (CFAR detection algorithms)
#   - CFAR_RADAR_Waterfall_ChirpSync.py (Enhanced CFAR with synchronization)
#   - Range_Doppler_Processing.py (MTI and pulse cancellation)
#   - target_detection_dbfs.py (Advanced CFAR implementations)
#   - CW_RADAR_Waterfall.py (Continuous wave processing)
#
# Original CN0566 examples copyright (C) 2022-2024 Analog Devices, Inc.
# Jon Kraft's foundational work provides the core radar processing algorithms
# that make this enhanced system possible.
#
# Enhanced features include:
#   - Advanced micro-Doppler drone classification
#   - Multi-pulse MTI processing for moving target indication  
#   - Enhanced Kalman tracking with threat assessment
#   - Adaptive CFAR detection with multiple methods
#   - Real-time 4D visualization and analysis
#   - MAVLink integration for autopilot systems
#
# All rights reserved. See individual source files for detailed licensing.
# ==============================================================================

# Laptop-Side Radar Processing Dashboard with Raw I/Q Reception

        # Ping the Pi to measure latency
        # on wifi, use the Pi's IP address: 192.168.0.7
        # your own address on the Pi is: 192.168.0.6
        # on hotspot, use the Pi's IP address: 192.168.137.62
        # your own address on the pi is: 192.168.137.1
        # pi ethernet: 192.168.100.2
        # laptop ethernet: 192.168.100.1

import socket
import threading
import struct
import pickle
import queue
import time
import numpy as np
from collections import deque
from scipy.signal.windows import chebwin
from scipy.ndimage import maximum_filter
import logging
import sys
import os
import signal
import atexit
import traceback
import random
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import pdist
from filterpy.kalman import KalmanFilter
from scipy.signal import stft

# Core dashboard imports
import pandas as pd
from dash.exceptions import PreventUpdate
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate
from flask import Flask
import plotly.graph_objs as go
import psutil
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from logging.handlers import RotatingFileHandler

# GPU acceleration check (moved after logger setup)
GPU_ACCELERATION_AVAILABLE = False

# === LAPTOP PROCESSING CONFIGURATION ===
# Network Configuration - Simple and Direct like working version
LAPTOP_SERVER_PORT = 9999
LAPTOP_IP = '192.168.0.6'  # Laptop's IP address - same as Pi transmitter target
PI_IP = '192.168.0.7'      # Expected Pi IP address

IQ_DATA_QUEUE = queue.Queue(maxsize=10)
PROCESSED_DATA_LOCK = threading.Lock()

# === LOGGING SETUP WITH UNICODE SUPPORT ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RadarApp")
if logger.hasHandlers():
    logger.handlers.clear()
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler with UTF-8 encoding support
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
# Fix Windows console encoding for Unicode characters
if hasattr(console_handler.stream, 'reconfigure'):
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except:
        pass
logger.addHandler(console_handler)

# File handler with UTF-8 encoding
file_handler = RotatingFileHandler('radar_dashboard.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# NOW add the GPU check and CPU optimization AFTER logger is defined
try:
    import pyopencl
    GPU_ACCELERATION_AVAILABLE = True
    logger.info("Intel GPU acceleration available")
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    logger.info("Install pyopencl for GPU acceleration: pip install pyopencl")

# Initialize Intel CPU optimizations
def optimize_intel_cpu():
    """Intel i5-1230U specific optimizations."""
    import os
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['NUMEXPR_MAX_THREADS'] = '10'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    try:
        import numpy as np
        np.seterr(all='ignore')
    except:
        pass

# Call optimization after it's defined
optimize_intel_cpu()

# === CFAR MODULE - ENHANCED IMPORT ===
CFAR_AVAILABLE = False
try:
    import importlib
    cfar_module = importlib.import_module('target_detection_dbfs')
    cfar = cfar_module.cfar
    CFAR_AVAILABLE = True
    logger.info("[SUCCESS] Advanced CFAR module loaded successfully")
except (ImportError, ModuleNotFoundError, AttributeError):
    CFAR_AVAILABLE = False
    logger.info("[WARNING] CFAR module not found. Using robust internal threshold detection.")
    
    # Enhanced fallback CFAR implementation
    def cfar(X_k, num_guard_cells=2, num_ref_cells=8, bias=15, cfar_method='average', fa_rate=0.2):
        """Enhanced fallback CFAR implementation with multiple detection methods"""
        N = X_k.size
        cfar_values = np.full(X_k.shape, np.min(X_k))
        
        for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
            min_index = center_index - (num_guard_cells + num_ref_cells)
            min_guard = center_index - num_guard_cells 
            max_index = center_index + (num_guard_cells + num_ref_cells) + 1
            max_guard = center_index + num_guard_cells + 1

            lower_nearby = X_k[min_index:min_guard]
            upper_nearby = X_k[max_guard:max_index]

            if cfar_method == 'average':
                mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
                cfar_values[center_index] = mean + bias
            elif cfar_method == 'greatest':
                mean = max(np.mean(lower_nearby), np.mean(upper_nearby))
                cfar_values[center_index] = mean + bias
            elif cfar_method == 'smallest':
                mean = min(np.mean(lower_nearby), np.mean(upper_nearby))
                cfar_values[center_index] = mean + bias
            elif cfar_method == 'false_alarm':
                refs = np.concatenate((lower_nearby, upper_nearby))
                noise_variance = np.sum(refs**2 / refs.size)
                cfar_values[center_index] = (noise_variance * -2 * np.log(fa_rate))**0.5
                
        targets_only = np.copy(X_k)
        targets_only[np.abs(X_k) <= np.abs(cfar_values)] = np.min(X_k)
        
        return cfar_values, targets_only

def receive_data_with_header(sock):
    try:
        size_data = b''
        while len(size_data) < 4:
            chunk = sock.recv(4 - len(size_data))
            if not chunk: return None
            size_data += chunk
        data_size = struct.unpack('>I', size_data)[0]
        
        received_data = b''
        while len(received_data) < data_size:
            chunk = sock.recv(min(4096, data_size - len(received_data)))
            if not chunk: return None
            received_data += chunk
        return pickle.loads(received_data)
    except Exception as e:
        print(f"[LAPTOP] Receive error: {e}")
        return None

# ==== SECTION: Global Constants & Configuration ====
CONFIG = {
    'hardware': {
        'SAMPLE_RATE': 2e6,
        'CENTER_FREQ': 2.1e9,
        'OUTPUT_FREQ': 10.25e9,
        'TX_WAVEFORM_SAMPLES': 2**14,
        'CHIRP_BW': 500e6,
        'RAMP_TIME': 500,
        'NUM_CHIRPS': 4,                      # CHANGED: From 32 to 4, as there are 4 full chirps in the 4096-sample buffer
        'SAMPLES_PER_CHIRP': 1000,            # CHANGED: From 128 to 1000, the true number of samples per 500µs ramp
        'NUM_AZ_ELEMENTS': 8,
        'ELEMENT_SPACING': 0.014,
        'DEFAULT_RX_GAIN': 40,
        'RPI_IP': os.getenv('RPI_IP', "ip:phaser.local"),
        'SDR_IP': "ip:192.168.2.1",
        'TDD_CYCLES_PER_US': 61.44,
    },
    'scanning': {
        'AZ_FOV': (-60, 60),
        'NUM_STEPS': 121,
        'VERTICAL_BEAMWIDTH': 60.0
    },
    'signal_processing': {
        'SIDELOBE_DB': 50,
        'CFAR_BIAS': 15,
        'MAX_RANGE_DISPLAY': 25.0,
        'MIN_RANGE_DISPLAY': 0.0,
        'MIN_SNR_THRESHOLD': -130,
        'FRAME_ACCUMULATION': 5,
        'DECAY_FACTOR': 0.95,
    },
    'dashboard': {
        'MAX_POINTS': 4000,
        'POINTS_TTL_SCANS': 75,
        'WATERFALL_ROWS': 64,
        'FAST_INTERVAL': 1500,
        'SLOW_INTERVAL': 2000,
    },
    'advanced_fmcw': {
        # Proven FMCW parameters from CN0566 examples
        'TRIANGULAR_RAMP_MODE': 'continuous_triangular',  # From FMCW_Velocity_RADAR_Waterfall.py
        'SAWTOOTH_BURST_MODE': 'single_sawtooth_burst',   # From FMCW_RADAR_Waterfall_ChirpSync.py
        'DELAY_WORD': 4095,                               # 12-bit delay word from examples
        'DELAY_CLK': 'PFD',                               # Clock source for delay
        'TDD_CYCLES_PER_US': 61.44,                       # TDD timing from examples
        'OPTIMAL_SAMPLE_RATES': [0.6e6, 2e6, 5e6],       # Tested sample rates
        'PROVEN_OUTPUT_FREQS': [10e9, 12.145e9],          # Tested frequencies
        'BLACKMAN_TAPER': [8, 34, 84, 127, 127, 84, 34, 8], # Proven antenna taper
        'SIGNAL_FREQ_IF': 100e3,                          # Standard IF frequency
        'ELEMENT_SPACING_M': 0.014,                       # 14mm element spacing
    },
    'system': {
        'WATCHDOG_TIMEOUT': 10.0,
        'MAX_RETRIES': 3,
    },
    'performance': {
    'MAX_DENSE_POINTS': 3000,  # Reduced from unlimited
    'KALMAN_HISTORY_LIMIT': 10,  # Limit tracker history
    'POINT_CLOUD_MEMORY_LIMIT_MB': 80,  # Memory budget
    'CPU_THRESHOLD_HIGH': 75,  # High CPU usage threshold
    'CPU_THRESHOLD_MEDIUM': 60,  # Medium CPU usage threshold
    'MEMORY_THRESHOLD_HIGH': 80,  # High memory usage threshold
    'MEMORY_THRESHOLD_MEDIUM': 70,  # Medium memory usage threshold
    },
}

# Enhanced profiles based on Jon Kraft's testing experience
INDOOR_PROFILE = {
    "label": "indoor",
    "MIN_PEAK_HEIGHT_DB": -60.0,
    "PEAK_DISTANCE": 25,
    "MIN_POINTS_FOR_DETECTION": 20,
    "DBSCAN_EPS": 0.35, # CHANGED: Reduced from 0.5 to better match 0.3m resolution
    "MIN_SAMPLES_IN_CLUSTER": 6,
    "TRACK_MAX_AGE_SECONDS": 2.0,
    "STATIC_TRACK_MAX_AGE_SECONDS": 60.0,
    "MAX_VALID_RANGE": 8.0,
    "FMCW_MODE": "continuous_triangular",
    "MTI_FILTER": "2pulse"
}

OUTDOOR_PROFILE = {
    "label": "outdoor",
    "MIN_PEAK_HEIGHT_DB": -70.0,
    "PEAK_DISTANCE": 40,
    "MIN_POINTS_FOR_DETECTION": 15,
    "DBSCAN_EPS": 0.35, # CHANGED: Reduced from 0.5 to better match 0.3m resolution
    "MIN_SAMPLES_IN_CLUSTER": 5,
    "TRACK_MAX_AGE_SECONDS": 3.0,
    "STATIC_TRACK_MAX_AGE_SECONDS": 120.0,
    "MAX_VALID_RANGE": 50.0,
    "FMCW_MODE": "single_sawtooth_burst",
    "MTI_FILTER": "3pulse"
}

# --- Resolve RPI_IP if using mDNS ---
if CONFIG['hardware']['RPI_IP'] == "ip:phaser.local":
    try:
        resolved_ip = socket.gethostbyname('phaser.local')
        CONFIG['hardware']['RPI_IP'] = f"ip:{resolved_ip}"
        logger.info(f"Resolved phaser.local to {resolved_ip}")
    except socket.gaierror:
        fallback_ip = "192.168.2.2"
        CONFIG['hardware']['RPI_IP'] = f"ip:{fallback_ip}"
        logger.warning(f"mDNS resolution failed. Using fallback: {fallback_ip}")

# Precompute radar parameters
RADAR_QUEUE = queue.Queue()

c = 3e8
slope = CONFIG['hardware']['CHIRP_BW'] / (CONFIG['hardware']['RAMP_TIME'] * 1e-6)
range_fft_size = CONFIG['hardware']['SAMPLES_PER_CHIRP']
range_freq_axis_onesided = np.fft.rfftfreq(range_fft_size, 1.0 / CONFIG['hardware']['SAMPLE_RATE'])
range_m_axis = range_freq_axis_onesided * c / (2 * slope)

max_theoretical_range = c * CONFIG['hardware']['SAMPLE_RATE'] / (4 * slope)
range_m_axis = np.clip(range_m_axis, 0, max_theoretical_range)
logger.info(f"Range axis: 0 to {range_m_axis[-1]:.2f}m, Resolution: {range_m_axis[1]:.3f}m")

PRI_s = (CONFIG['hardware']['RAMP_TIME'] * 1e-6) + 1.5e-3
doppler_fft_size = CONFIG['hardware']['NUM_CHIRPS']
doppler_freq_axis_shifted = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, PRI_s))
velocity_axis = doppler_freq_axis_shifted * c / (2 * CONFIG['hardware']['OUTPUT_FREQ'])

azimuth_angles_scan = np.linspace(CONFIG['scanning']['AZ_FOV'][0], CONFIG['scanning']['AZ_FOV'][1], CONFIG['scanning']['NUM_STEPS'])
az_angles_beampattern = np.linspace(-90, 90, 181)

# === LAPTOP-ONLY GLOBAL VARIABLES ===
iq_tx_if_waveform: Optional[np.ndarray] = None
HARDWARE_AVAILABLE = False

RADAR_DATA_LOCK = threading.Lock()
RADAR_DATA = {
    'persistent_points': {angle: [] for angle in azimuth_angles_scan},
    'range_doppler_map': np.full((doppler_fft_size, len(range_m_axis)), -200.0, dtype=float),
    'waterfall': deque(maxlen=CONFIG['dashboard']['WATERFALL_ROWS']),
    'beampattern': np.zeros(len(az_angles_beampattern)),
    'current_az': 0.0,
    'error': None,
    'data_is_ready': False,
    'last_update': 0.0,
    'loop_time': 0.0,
    'scan_progress': 0.0,
    'scan_count': 0,
    'cfar_bias': CONFIG['signal_processing']['CFAR_BIAS'],
    'rx_gain': CONFIG['hardware']['DEFAULT_RX_GAIN'],
    'active_profile_name': 'indoor',
    'last_heartbeat_time': time.time()
}

TRACKED_OBJECTS = {}
NEXT_OBJECT_ID = 0

# ==== SECTION: Helper, Processing & Hardware Functions ====
def classify_semantic_object(cluster_df, environment_mode):
    """
    UPGRADED FOR 500 MHz: Adjusted thresholds for improved 0.3m range resolution
    """
    if len(cluster_df) < 5:  # RELAXED: Back to 5 for 500 MHz precision
        return None
        
    centroid = cluster_df[['x', 'y', 'z']].mean().values
    range_val = np.linalg.norm(centroid)
    avg_velocity = cluster_df['velocity'].mean()
    velocity_std = cluster_df['velocity'].std()
    avg_snr = cluster_df['snr'].mean()
    point_count = len(cluster_df)
    
    # RELAXED: More appropriate for 500 MHz resolution
    if velocity_std > 0.3:  # Back to more reasonable threshold
        return None
    
    # RELAXED: More appropriate SNR threshold for 500 MHz
    if avg_snr < -80:  # More lenient SNR threshold
        return None
    
    # Check for micro-Doppler drone signature (if exists)
    has_micro_doppler = False
    if 'micro_doppler_type' in cluster_df.columns:
        has_micro_doppler = (cluster_df['micro_doppler_type'] == 'DRONE_CONFIRMED').any()
    
    if has_micro_doppler:
        # CONFIRMED DRONE with micro-Doppler evidence
        blade_freq = cluster_df['blade_frequency'].mean() if 'blade_frequency' in cluster_df.columns else 0
        
        return {
            'type': 'DRONE_THREAT',
            'subtype': 'Confirmed Drone',
            'position': centroid,
            'range': range_val,
            'velocity': avg_velocity,
            'confidence': 0.95,
            'threat_level': 'HIGH',
            'properties': {'blade_frequency': blade_freq, 'point_count': point_count, 'snr': avg_snr},
            'render_priority': 1,
            'timestamp': time.time()
        }
    
    # RELAXED for 500 MHz: Smaller clusters are normal with better resolution
    elif abs(avg_velocity) < 0.05 and velocity_std < 0.02:  # Slightly more lenient
        if point_count > 12 and range_val > 1.0 and avg_snr > -65:  # RELAXED: Smaller point count, closer range, more lenient SNR
            return {
                'type': 'STATIC_OBJECT',
                'subtype': 'Large Structure',
                'position': centroid,
                'range': range_val,
                'velocity': avg_velocity,
                'confidence': min(1.0, point_count / 15.0),  # Adjusted for smaller clusters
                'threat_level': 'OBSTACLE',
                'properties': {'point_count': point_count, 'snr': avg_snr},
                'render_priority': 3,
                'timestamp': time.time()
            }
        elif point_count > 8 and avg_snr > -70:  # RELAXED: Much lower requirements for medium objects
            return {
                'type': 'STATIC_OBJECT',
                'subtype': 'Medium Structure',
                'position': centroid,
                'range': range_val,
                'velocity': avg_velocity,
                'confidence': min(1.0, point_count / 12.0),  # Adjusted for smaller clusters
                'threat_level': 'OBSTACLE',
                'properties': {'point_count': point_count, 'snr': avg_snr},
                'render_priority': 3,
                'timestamp': time.time()
            }
    
    # RELAXED moving object detection for 500 MHz
    elif abs(avg_velocity) > 0.15 and velocity_std < 0.2:  # RELAXED: Slightly lower velocity threshold
        if point_count > 6 and avg_snr > -75:  # RELAXED: Lower point count and more lenient SNR
            return {
                'type': 'MOVING_OBJECT',
                'subtype': 'Person/Vehicle',
                'position': centroid,
                'range': range_val,
                'velocity': avg_velocity,
                'confidence': min(1.0, (avg_snr + 85) / 20.0),  # Adjusted confidence calculation
                'threat_level': 'DYNAMIC',
                'properties': {'point_count': point_count, 'snr': avg_snr, 'velocity_std': velocity_std},
                'render_priority': 2,
                'timestamp': time.time()
            }
    
    # Reject everything else to reduce false positives
    return None

def create_semantic_legend():
    """Create semantic object color coding legend positioned on the right side."""
    legend_traces = []
    
    # Updated semantic colors matching classify_semantic_object function
    semantic_colors = {
        'DRONE THREAT': '#FF0000',        # Red - High priority
        'MOVING OBJECT': '#00FF00',       # Green - Dynamic targets
        'STATIC OBJECT': '#FFA500',       # Orange - Combined static objects
        'DETECTION': '#00FFFF',           # Cyan - Generic detections
        'UNKNOWN': '#8A2BE2'              # Purple - Unclassified
    }
    
    # Position on RIGHT side to avoid voxel legend overlap
    x_pos = 22  # Right side positioning
    y_pos = 1.1
    for label, color in semantic_colors.items():
        legend_traces.append(go.Scatter3d(
            x=[x_pos], y=[20], z=[y_pos * 10],
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='square', 
                       line=dict(width=2, color='white')),
            text=[f"■ {label}"],
            textposition="middle right",
            textfont=dict(color='white', size=10),
            showlegend=False,
            hoverinfo='none',
            name=''
        ))
        y_pos -= 0.15
    
    return legend_traces

def create_micro_doppler_legend():
    """Create micro-Doppler legend positioned on the LEFT side to avoid overlap."""
    legend_traces = []
    
    # Micro-Doppler indicators
    micro_doppler_colors = {
        'Drone Blades': '#FFFF00',        # Yellow rings for blade signatures
        'Fan Motion': '#FF69B4',          # Hot pink for fan signatures
        'Rotor Activity': '#FFA500'       # Orange for general rotor activity
    }
    
    # Position on RIGHT side to avoid semantic legend overlap
    x_pos = 22  # Right side positioning
    y_pos = 0.2
    
    # Add micro-Doppler legend header
    legend_traces.append(go.Scatter3d(
        x=[x_pos], y=[18], z=[y_pos * 10],
        mode='text',
        text=["--- Micro-Doppler ---"],
        textfont=dict(color='yellow', size=12, family='bold'),
        showlegend=False,
        hoverinfo='none'
    ))
    y_pos -= 0.15
    
    for label, color in micro_doppler_colors.items():
        legend_traces.append(go.Scatter3d(
            x=[x_pos], y=[20], z=[y_pos * 10],
            mode='markers+text',
            marker=dict(size=8, color=color, symbol='circle',
                       line=dict(width=3, color=color)),
            text=[f"○ {label}"],
            textposition="middle right",
            textfont=dict(color='white', size=10),
            showlegend=False,
            hoverinfo='none',
            name=''
        ))
        y_pos -= 0.12
    
    return legend_traces

def create_cluster_box_legend():
    """Create a legend for Kalman tracker bounding box colors."""
    legend_traces = []
    
    # Colors are based on the KalmanTracker class and CONFIRMED_COLORS dict
    cluster_colors = {
        'Tracking...': 'rgb(128, 128, 128)', # Preliminary color
        'Drone/UAV': 'magenta',
        'Person': 'red',
        'Moving Object': 'lime',
        'Wall/Structure': 'cyan',
        'Furniture': 'orange',
        'Small Object': 'yellow'
    }
    
    x_pos = -22 # Position on the LEFT side
    y_pos = 0.9

    legend_traces.append(go.Scatter3d(
        x=[x_pos], y=[20], z=[y_pos * 15], mode='text',
        text=["--- Cluster Status ---"],
        textfont=dict(color='lime', size=12, family='bold'),
        showlegend=False, hoverinfo='none'
    ))
    y_pos -= 0.15

    for label, color in cluster_colors.items():
        legend_traces.append(go.Scatter3d(
            x=[x_pos], y=[20], z=[y_pos * 10], mode='markers+text',
            marker=dict(size=4, color=color, symbol='square-open'),
            text=[f"□ {label}"],
            textposition="middle right",
            textfont=dict(color='white', size=10),
            showlegend=False, hoverinfo='none', name=''
        ))
        y_pos -= 0.12
        
    return legend_traces    

def remove_statistical_outliers(df, z_threshold=3):
    """Remove points that are statistical outliers in range/velocity space"""
    if len(df) < 10: return df
    df_copy = df.copy()
    for col in ['range', 'velocity']:
        if df_copy[col].std() == 0: continue
        z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
        df_copy = df_copy[z_scores < z_threshold]
    return df_copy

def apply_spatial_filtering(df, grid_size=0.2):
    """Apply spatial filtering (voxel grid) to average points and reduce noise."""
    if len(df) < 5: return df
    df_copy = df.copy()
    df_copy['x_grid'] = np.round(df_copy['x'] / grid_size)
    df_copy['y_grid'] = np.round(df_copy['y'] / grid_size)
    df_copy['z_grid'] = np.round(df_copy['z'] / grid_size)
    return df_copy.groupby(['x_grid', 'y_grid', 'z_grid']).agg(
        x=('x', 'mean'), y=('y', 'mean'), z=('z', 'mean'),
        range=('range', 'mean'), velocity=('velocity', 'mean'),
        snr=('snr', 'max'), timestamp=('timestamp', 'first')
    ).reset_index()

def create_range_rings(max_range, num_rings=4, color='rgba(100,100,100,0.3)'):
    """Creates range rings on the XY plane for better depth perception."""
    traces = []
    angles = np.linspace(0, 2 * np.pi, 100)
    for i in range(1, num_rings + 1):
        radius = (max_range / num_rings) * i
        x_ring = radius * np.cos(angles)
        y_ring = radius * np.sin(angles)
        traces.append(go.Scatter3d(x=x_ring, y=y_ring, z=np.zeros_like(x_ring), mode='lines',
                                   line=dict(color=color, width=1, dash='dash'), hoverinfo='none', name=''))
    return traces

def calculate_centroid(points_df):
    """Calculates the XYZ center of a DataFrame of points."""
    return points_df[['x', 'y', 'z']].mean().values

def classify_cluster(cluster_df, bbox_params):
    """ENHANCED: Returns more detailed classification with confidence."""
    try:
        point_count = len(cluster_df)
        if point_count < 5: return "Noise"

        dims = bbox_params.get('dimensions', [0, 0, 0])
        length, width, height = sorted(dims, reverse=True)
        avg_velocity = cluster_df['velocity'].mean() if 'velocity' in cluster_df.columns else 0.0
        velocity_std_dev = cluster_df['velocity'].std() if point_count > 1 else 0.0
        velocity_std_dev = 0.0 if pd.isna(velocity_std_dev) else velocity_std_dev
        avg_snr = cluster_df['snr'].mean() if 'snr' in cluster_df.columns else -120.0
        
        # Calculate spatial density
        volume = length * width * height if length > 0 and width > 0 and height > 0 else 0.001
        density = point_count / volume

        # Enhanced classification logic
        if abs(avg_velocity) < 0.05:  # Stationary objects
            if length > 2.0 and height > 1.5:
                return "Wall/Large Surface"
            elif 0.5 < length < 2.0 and height > 1.0:
                return "Furniture/Cabinet"
            elif length < 1.0 and width < 1.0:
                return "Small Object"
            else:
                return "Static Structure"
        
        elif abs(avg_velocity) > 0.2:  # Moving objects
            if (0.1 < length < 1.2) and (point_count > 8) and (velocity_std_dev < 0.4) and (avg_snr > -65):
                return "Drone/UAV"
            elif (0.8 < height < 2.0) and (length < 1.5) and (point_count > 15):
                return "Person"
            else:
                return "Moving Object"
        
        return "Unknown Object"
        
    except Exception as e:
        logger.error(f"Classifier failed: {e}\n{traceback.format_exc()}")
        return "Classification Error"
    
# ==== SECTION: Kalman Tracker Class ====
from filterpy.kalman import KalmanFilter

class EnhancedKalmanTracker:
    """
    Enhanced Kalman Filter-based tracker with drone-specific capabilities
    Features: 
    - Adaptive noise models for different target types
    - Micro-Doppler signature tracking
    - Threat level assessment
    - Maneuver detection
    """
    def __init__(self, initial_measurement, track_id, target_type='UNKNOWN'):
        self.id = track_id
        self.target_type = target_type
        self.kf = KalmanFilter(dim_x=9, dim_z=3)  # Extended state for acceleration
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        # Measurement vector: [x, y, z]

        # --- Define the State Transition Matrix (F) ---
        dt = 0.1  # Time step (matches our 10Hz MAVLink exporter)
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Acceleration persists
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # --- Define the Measurement Function (H) ---
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        # --- Adaptive Noise Matrices based on target type ---
        self._configure_noise_matrices(target_type)

        # --- Initial State ---
        self.kf.x[:3] = initial_measurement.reshape(3, 1)
        
        # --- Enhanced tracking properties ---
        self.history = [self.kf.x.copy()]
        self.last_seen = time.time()
        self.hits = 1
        self.misses = 0
        self.is_confirmed = False
        self.label = self._get_initial_label(target_type)
        self.color = self._get_target_color(target_type)
        self.classification_history = deque(maxlen=10)
        
        # Drone-specific properties
        self.micro_doppler_signatures = deque(maxlen=20)
        self.threat_level = 'LOW'
        self.maneuver_detected = False
        self.last_maneuver_time = 0
        self.velocity_history = deque(maxlen=15)
        self.confidence_score = 0.5

    def _configure_noise_matrices(self, target_type):
        """Configure noise matrices based on target type"""
        if target_type in ['DRONE_CONFIRMED', 'DRONE_PROBABLE']:
            # Drones: Higher maneuverability, moderate measurement trust
            self.kf.R = np.eye(3) * 5.0  # Moderate measurement noise
            self.kf.Q = np.eye(9) * 0.5  # Higher process noise for maneuvers
            self.kf.Q[6:9, 6:9] *= 2.0   # Even higher for acceleration
        elif target_type == 'STATIC_OBJECT':
            # Static objects: Very low process noise, high measurement trust
            self.kf.R = np.eye(3) * 2.0
            self.kf.Q = np.eye(9) * 0.01
        else:
            # Unknown targets: Balanced approach
            self.kf.R = np.eye(3) * 8.0
            self.kf.Q = np.eye(9) * 0.2

    def _get_initial_label(self, target_type):
        """Get initial label based on target type"""
        labels = {
            'DRONE_CONFIRMED': 'DRONE',
            'DRONE_PROBABLE': 'LIKELY DRONE',
            'DRONE_POSSIBLE': 'POSSIBLE DRONE',
            'MOVING_OBJECT': 'MOVING',
            'STATIC_OBJECT': 'STATIC',
            'UNKNOWN': 'UNKNOWN'
        }
        return labels.get(target_type, 'OBJECT')

    def _get_target_color(self, target_type):
        """Get color based on target type and threat level"""
        colors = {
            'DRONE_CONFIRMED': 'rgb(255, 0, 0)',    # Red - High threat
            'DRONE_PROBABLE': 'rgb(255, 165, 0)',   # Orange - Medium threat
            'DRONE_POSSIBLE': 'rgb(255, 255, 0)',   # Yellow - Low threat
            'MOVING_OBJECT': 'rgb(0, 255, 0)',      # Green - Moving
            'STATIC_OBJECT': 'rgb(128, 128, 128)',  # Gray - Static
            'UNKNOWN': 'rgb(0, 0, 255)'             # Blue - Unknown
        }
        return colors.get(target_type, 'rgb(128, 128, 128)')

    def predict(self):
        """Enhanced prediction with maneuver detection"""
        self.kf.predict()
        
        # Update velocity history for maneuver detection
        current_velocity = self.velocity.copy()
        self.velocity_history.append(current_velocity)
        
        # Detect sudden maneuvers
        if len(self.velocity_history) >= 5:
            self._detect_maneuvers()

    def update(self, measurement, micro_doppler_data=None):
        """Enhanced update with micro-Doppler integration"""
        self.kf.update(measurement.reshape(3, 1))
        self.history.append(self.kf.x.copy())
        self.last_seen = time.time()
        self.hits += 1
        self.misses = 0  # Reset miss counter
        
        # Update micro-Doppler signatures
        if micro_doppler_data:
            self.micro_doppler_signatures.append(micro_doppler_data)
            self._update_classification(micro_doppler_data)
        
        # Update threat assessment
        self._assess_threat_level()

    def update_miss(self):
        """Handle missed detections"""
        self.misses += 1
        # Continue prediction but don't update with measurements

    def _detect_maneuvers(self):
        """Detect sudden direction or speed changes"""
        if len(self.velocity_history) < 5:
            return
        
        recent_velocities = np.array(list(self.velocity_history)[-5:])
        
        # Calculate velocity changes
        velocity_changes = np.diff(recent_velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(velocity_changes, axis=1)
        
        # Detect sudden acceleration (maneuver)
        maneuver_threshold = 2.0  # m/s²
        if np.any(acceleration_magnitudes > maneuver_threshold):
            current_time = time.time()
            if current_time - self.last_maneuver_time > 2.0:  # Avoid multiple detections
                self.maneuver_detected = True
                self.last_maneuver_time = current_time
                logger.info(f"Maneuver detected for track {self.id}")

    def _update_classification(self, micro_doppler_data):
        """Update target classification based on micro-Doppler data"""
        if 'type' in micro_doppler_data:
            self.classification_history.append(micro_doppler_data['type'])
            
            # Update target type based on consistent classifications
            if len(self.classification_history) >= 3:
                recent_classifications = list(self.classification_history)[-3:]
                drone_classifications = [c for c in recent_classifications 
                                       if 'DRONE' in c]
                
                if len(drone_classifications) >= 2:
                    # Consistent drone classification
                    if self.target_type not in ['DRONE_CONFIRMED', 'DRONE_PROBABLE']:
                        self.target_type = 'DRONE_PROBABLE'
                        self.label = 'LIKELY DRONE'
                        self.color = 'rgb(255, 165, 0)'

    def _assess_threat_level(self):
        """Assess threat level based on multiple factors"""
        threat_score = 0.0
        
        # Factor 1: Target type
        if self.target_type == 'DRONE_CONFIRMED':
            threat_score += 0.4
        elif self.target_type == 'DRONE_PROBABLE':
            threat_score += 0.3
        elif self.target_type == 'DRONE_POSSIBLE':
            threat_score += 0.2
        
        # Factor 2: Speed and maneuverability
        current_speed = np.linalg.norm(self.velocity)
        if 1.0 < current_speed < 20.0:  # Typical drone speed range
            threat_score += 0.2
        
        if self.maneuver_detected:
            threat_score += 0.1
        
        # Factor 3: Proximity (closer = higher threat)
        distance = np.linalg.norm(self.position)
        if distance < 10.0:
            threat_score += 0.2
        elif distance < 25.0:
            threat_score += 0.1
        
        # Factor 4: Micro-Doppler confidence
        if len(self.micro_doppler_signatures) > 0:
            avg_confidence = np.mean([sig.get('confidence', 0) 
                                    for sig in self.micro_doppler_signatures])
            threat_score += avg_confidence * 0.1
        
        # Determine threat level
        if threat_score > 0.7:
            self.threat_level = 'HIGH'
            self.color = 'rgb(255, 0, 0)'  # Red
        elif threat_score > 0.4:
            self.threat_level = 'MEDIUM'
            self.color = 'rgb(255, 165, 0)'  # Orange
        else:
            self.threat_level = 'LOW'
        
        self.confidence_score = min(1.0, threat_score)

    @property
    def position(self):
        return self.kf.x[:3].flatten()

    @property
    def velocity(self):
        return self.kf.x[3:6].flatten()

    @property
    def acceleration(self):
        return self.kf.x[6:9].flatten()

    def get_predicted_position(self, dt_future):
        """Predict position at future time"""
        pos = self.position
        vel = self.velocity
        acc = self.acceleration
        return pos + vel * dt_future + 0.5 * acc * dt_future**2

    def is_stale(self, max_age_seconds=5.0):
        """Check if track is too old"""
        return (time.time() - self.last_seen) > max_age_seconds

    def should_delete(self):
        """Determine if track should be deleted"""
        return (self.misses > 5 or  # Too many misses
                self.is_stale(10.0) or  # Too old
                (self.hits < 3 and self.misses > 2))  # Poor track quality

# Maintain backward compatibility
class KalmanTracker(EnhancedKalmanTracker):
    """Legacy wrapper for backward compatibility"""
    def __init__(self, initial_measurement, track_id):
        super().__init__(initial_measurement, track_id, 'UNKNOWN')

    @property
    def velocity(self):
        return self.kf.x[3:].flatten()

def apply_spatial_filtering(df, grid_size=0.2):
    """Applies a voxel grid filter to downsample the point cloud and reduce noise."""
    if len(df) < 5: return df
    df_copy = df.copy()
    # Create a unique integer ID for each voxel
    df_copy['x_grid'] = (df_copy['x'] // grid_size).astype(int)
    df_copy['y_grid'] = (df_copy['y'] // grid_size).astype(int)
    df_copy['z_grid'] = (df_copy['z'] // grid_size).astype(int)
    # Get the point with the highest SNR from each populated voxel
    return df_copy.loc[df_copy.groupby(['x_grid', 'y_grid', 'z_grid'])['snr'].idxmax()]

def intelligent_point_sampling(df, max_points=4000):
    """Intelligently samples points, prioritizing those with higher SNR."""
    if len(df) <= max_points: return df
    # Prioritize keeping 80% of the display budget for the top 25% of signals
    high_snr_threshold = df['snr'].quantile(0.75)
    df_high_snr = df[df['snr'] >= high_snr_threshold]
    df_low_snr = df[df['snr'] < high_snr_threshold]
    high_snr_budget = min(len(df_high_snr), int(max_points * 0.8))
    low_snr_budget = max_points - high_snr_budget
    df_sampled_high = df_high_snr.sample(n=high_snr_budget)
    df_sampled_low = df_low_snr.sample(n=min(low_snr_budget, len(df_low_snr)))
    return pd.concat([df_sampled_high, df_sampled_low], ignore_index=True)

def coherent_multipath_suppression(range_doppler_history: List[np.ndarray],
                                  coherence_threshold: float = 0.7) -> np.ndarray:
    """Suppress multipath using multi-frame coherence"""
    if len(range_doppler_history) < 3:
        return range_doppler_history[-1] if range_doppler_history else np.array([])

    frames = np.stack(range_doppler_history[-3:], axis=0)
    coherence_map = np.zeros_like(frames[0])

    for i in range(frames.shape[1]):
        for j in range(frames.shape[2]):
            pixel_history = frames[:, i, j]
            if np.std(pixel_history) > 0:
                coherence = np.corrcoef(pixel_history, np.arange(len(pixel_history)))[0, 1]
                coherence_map[i, j] = abs(coherence) if not np.isnan(coherence) else 0

    coherent_frame = frames[-1].copy()
    coherent_frame[coherence_map < coherence_threshold] *= 0.3
    return coherent_frame

def enhanced_mti_processing(radar_data_history: List[np.ndarray], mti_mode='3pulse') -> np.ndarray:
    """
    Enhanced Moving Target Indicator (MTI) processing with 2-pulse and 3-pulse cancellation
    Inspired by Range_Doppler_Processing.py for better drone detection
    """
    if len(radar_data_history) < 2:
        return radar_data_history[-1] if radar_data_history else np.array([])
    
    current_frame = radar_data_history[-1]
    
    if mti_mode == '2pulse' and len(radar_data_history) >= 2:
        # 2-pulse canceller MTI
        prev_frame = radar_data_history[-2]
        
        # Estimate phase correlation between frames
        correlation = np.mean(current_frame * np.conj(prev_frame))
        phase_diff = np.angle(correlation)
        
        # Apply phase compensation and subtraction
        mti_frame = current_frame - prev_frame * np.exp(-1j * phase_diff)
        return mti_frame
        
    elif mti_mode == '3pulse' and len(radar_data_history) >= 3:
        # 3-pulse canceller MTI for enhanced static clutter rejection
        frame_t = radar_data_history[-1]
        frame_t1 = radar_data_history[-2] 
        frame_t2 = radar_data_history[-3]
        
        # First stage: 2-pulse cancellation
        correlation_1 = np.mean(frame_t * np.conj(frame_t1))
        phase_diff_1 = np.angle(correlation_1)
        stage1_t = frame_t - frame_t1 * np.exp(-1j * phase_diff_1)
        
        correlation_2 = np.mean(frame_t1 * np.conj(frame_t2))
        phase_diff_2 = np.angle(correlation_2)
        stage1_t1 = frame_t1 - frame_t2 * np.exp(-1j * phase_diff_2)
        
        # Second stage: Apply another 2-pulse cancellation
        correlation_3 = np.mean(stage1_t * np.conj(stage1_t1))
        phase_diff_3 = np.angle(correlation_3)
        mti_frame = stage1_t - stage1_t1 * np.exp(-1j * phase_diff_3)
        
        return mti_frame
    else:
        # Default: simple frame differencing
        if len(radar_data_history) >= 2:
            return radar_data_history[-1] - radar_data_history[-2]
        else:
            return current_frame

def advanced_cfar_detection(range_doppler_magnitude: np.ndarray, 
                          num_guard_cells=2, num_ref_cells=8, 
                          cfar_bias=15, cfar_method='average') -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced CFAR detection with multiple methods for enhanced target detection
    """
    if not CFAR_AVAILABLE:
        # Use enhanced fallback
        return cfar(range_doppler_magnitude.flatten(), num_guard_cells, num_ref_cells, 
                   cfar_bias, cfar_method), range_doppler_magnitude
    
    # Apply CFAR detection to each range bin
    cfar_threshold = np.zeros_like(range_doppler_magnitude)
    detected_targets = np.zeros_like(range_doppler_magnitude)
    
    for range_idx in range(range_doppler_magnitude.shape[0]):
        range_profile = range_doppler_magnitude[range_idx, :]
        
        try:
            cfar_values, targets_only = cfar(range_profile, num_guard_cells, 
                                           num_ref_cells, cfar_bias, cfar_method)
            cfar_threshold[range_idx, :] = cfar_values
            detected_targets[range_idx, :] = targets_only
        except Exception as e:
            logger.warning(f"CFAR failed for range bin {range_idx}: {e}")
            # Fallback to simple threshold
            threshold = np.mean(range_profile) + cfar_bias
            cfar_threshold[range_idx, :] = threshold
            detected_targets[range_idx, :] = np.where(range_profile > threshold, 
                                                    range_profile, 0)
    
    return cfar_threshold, detected_targets

def enhanced_freq_process(radar_data, min_scale=4, max_scale=6):
    """
    Enhanced frequency processing method from Range_Doppler_Processing.py
    Optimized for FMCW radar range-doppler map generation
    """
    try:
        # Proven 2D FFT approach
        rx_chirps_fft = np.fft.fftshift(np.abs(np.fft.fft2(radar_data)))
        range_doppler_data = np.log10(rx_chirps_fft + 1e-12).T  # Transpose for proper orientation
        
        # Proven clipping approach for visualization
        range_doppler_data = np.clip(range_doppler_data, min_scale, max_scale)
        
        return range_doppler_data
        
    except Exception as e:
        logger.error(f"Jon Kraft freq_process failed: {e}")
        # Fallback to basic processing
        return np.log10(np.abs(radar_data) + 1e-12)

def enhanced_pulse_canceller(radar_data, mti_filter='3pulse'):
    """
    Enhanced pulse cancellation method from Range_Doppler_Processing.py
    Enhanced for better moving target indication
    """
    if len(radar_data.shape) != 2:
        logger.warning("Invalid radar data shape for pulse canceller")
        return radar_data
        
    num_chirps, num_samples = radar_data.shape
    
    if mti_filter == '2pulse' and num_chirps >= 2:
        # Jon Kraft's 2-pulse canceller implementation
        Chirp2P = np.empty([num_chirps, num_samples], dtype=complex)
        
        for chirp in range(num_chirps-1):
            chirpI = radar_data[chirp, :]
            chirpI1 = radar_data[chirp+1, :]
            
            # Cross-correlation for phase alignment (Jon Kraft's method)
            chirp_correlation = np.correlate(chirpI, chirpI1, 'valid')
            if len(chirp_correlation) > 0:
                angle_diff = np.angle(chirp_correlation[0])
                Chirp2P[chirp, :] = chirpI1 - chirpI * np.exp(-1j * angle_diff)
            else:
                Chirp2P[chirp, :] = chirpI1 - chirpI
        
        return Chirp2P
        
    elif mti_filter == '3pulse' and num_chirps >= 3:
        # Jon Kraft's 3-pulse canceller for enhanced clutter rejection
        Chirp2P = np.empty([num_chirps, num_samples], dtype=complex)
        Chirp3P = np.empty([num_chirps, num_samples], dtype=complex)
        
        # First stage: 2-pulse cancellation
        for chirp in range(num_chirps-1):
            chirpI = radar_data[chirp, :]
            chirpI1 = radar_data[chirp+1, :]
            chirp_correlation = np.correlate(chirpI, chirpI1, 'valid')
            if len(chirp_correlation) > 0:
                angle_diff = np.angle(chirp_correlation[0])
                Chirp2P[chirp, :] = chirpI1 - chirpI * np.exp(-1j * angle_diff)
            else:
                Chirp2P[chirp, :] = chirpI1 - chirpI
        
        # Second stage: Apply another 2-pulse cancellation
        for chirp in range(num_chirps-2):
            chirpI = Chirp2P[chirp, :]
            chirpI1 = Chirp2P[chirp+1, :]
            Chirp3P[chirp, :] = chirpI1 - chirpI
        
        return Chirp3P
    
    else:
        # No MTI processing
        return radar_data

def enhanced_range_doppler_processing(iq_ch0, iq_ch1, use_mti=True, mti_filter='3pulse'):
    """
    Enhanced range-doppler processing combining Jon Kraft's proven methods
    with modern enhancements for better drone detection
    """
    try:
        num_chirps = CONFIG['hardware']['NUM_CHIRPS']
        samples_per_chirp = CONFIG['hardware']['SAMPLES_PER_CHIRP']
        
        # Reshape data into chirp matrix format
        if iq_ch0.size >= num_chirps * samples_per_chirp:
            ch0_matrix = iq_ch0[:num_chirps * samples_per_chirp].reshape((num_chirps, samples_per_chirp))
            ch1_matrix = iq_ch1[:num_chirps * samples_per_chirp].reshape((num_chirps, samples_per_chirp))
        else:
            logger.warning("Insufficient data for range-doppler processing")
            return None, None
        
        # Apply enhanced pulse cancellation for MTI
        if use_mti:
            ch0_matrix = enhanced_pulse_canceller(ch0_matrix, mti_filter)
            ch1_matrix = enhanced_pulse_canceller(ch1_matrix, mti_filter)
        
        # Enhanced frequency processing
        rd_map_ch0 = enhanced_freq_process(ch0_matrix)
        rd_map_ch1 = enhanced_freq_process(ch1_matrix)
        
        # Combine channels (average for better SNR)
        combined_rd_map = (rd_map_ch0 + rd_map_ch1) / 2.0
        
        return combined_rd_map, (rd_map_ch0, rd_map_ch1)
        
    except Exception as e:
        logger.error(f"Enhanced range-doppler processing failed: {e}")
        return None, None

def detect_environment_type(points_list: List[dict]) -> str:
    """Automatically detect indoor vs outdoor environment"""
    if not points_list:
        return 'unknown'

    ranges = [np.sqrt(p['x']**2 + p['y']**2 + p['z']**2) for p in points_list]
    short_range = sum(1 for r in ranges if r < 5)
    medium_range = sum(1 for r in ranges if 5 <= r < 15)
    long_range = sum(1 for r in ranges if r >= 15)

    total_detections = len(ranges)
    if total_detections == 0:
        return 'unknown'

    if short_range / total_detections > 0.6:
        return 'indoor'
    elif long_range / total_detections > 0.3:
        return 'outdoor'
    else:
        return 'transitional'

def detect_objects_dbscan(points_df, eps=0.5, min_samples=5):  # UPGRADED: eps reduced for 500 MHz bandwidth
    """Cluster points using DBSCAN and return a list of object dataframes."""
    if points_df.empty or len(points_df) < min_samples:
        return []

    coords = points_df[['x', 'y', 'z']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    clusters = []
    for label in set(labels):
        if label != -1:
            clusters.append(points_df[labels == label])
    return clusters

def calculate_oriented_bounding_box(cluster_df):
    """Calculate an oriented bounding box for a cluster of points using PCA."""
    coords = cluster_df[['x', 'y', 'z']].values
    if len(coords) < 3: return None

    pca = PCA(n_components=3)
    pca.fit(coords)

    center = pca.mean_
    projected = pca.transform(coords)
    min_coords = projected.min(axis=0)
    max_coords = projected.max(axis=0)
    dims = max_coords - min_coords

    return {'center': center, 'dimensions': dims, 'rotation_matrix': pca.components_}

SHOW_CLUSTER_BOXES = True  # Add this flag to control visibility

def create_bounding_box_trace(bbox_params: Dict, color: str = 'lime') -> go.Scatter3d:
    """Create bounding box wireframe trace."""
    if not SHOW_CLUSTER_BOXES:
        return None # Skip creating cluster boxes if the flag is off
    if not bbox_params:
        return None

    center = bbox_params['center']
    dims = bbox_params['dimensions']
    rotation_matrix = bbox_params['rotation_matrix']

    unit_corners = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])

    final_corners = (unit_corners * dims) @ rotation_matrix + center
    x, y, z = final_corners[:, 0], final_corners[:, 1], final_corners[:, 2]

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    x_lines, y_lines, z_lines = [], [], []
    for p1_idx, p2_idx in lines:
        x_lines.extend([x[p1_idx], x[p2_idx], None])
        y_lines.extend([y[p1_idx], y[p2_idx], None])
        z_lines.extend([z[p1_idx], z[p2_idx], None])

    return go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo='none'
    )

def compute_beampattern_plot_data(taper_gain: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Computes the full beampattern (gain vs. angle) for visualization."""
    num_elements = CONFIG['hardware']['NUM_AZ_ELEMENTS']
    rf_freq = CONFIG['hardware']['OUTPUT_FREQ']
    element_spacing = CONFIG['hardware']['ELEMENT_SPACING']
    c = 3e8
    
    viz_angles_rad = np.radians(az_angles_beampattern)
    applied_phase_rad = np.radians(phase_deg)
    
    array_factor = np.zeros(len(viz_angles_rad), dtype=np.complex64)
    for i, angle_rad in enumerate(viz_angles_rad):
        steering_vector = np.exp(-1j * 2 * np.pi * element_spacing / (c/rf_freq) * np.sin(angle_rad) * np.arange(num_elements))
        weights = taper_gain * np.exp(1j * applied_phase_rad)
        array_factor[i] = np.sum(weights * steering_vector)
        
    beampattern_db = 20 * np.log10(np.abs(array_factor) + 1e-12)
    return beampattern_db - np.max(beampattern_db)

def create_radar_cone_traces_with_opacity(max_range: float, az_fov_deg: Tuple[float, float], cone_opacity: float = 0.15, el_fov_deg: float = 60.0):
    """
    Enhanced radar cone with user-controllable opacity.
    """
    traces = []
    grid_color = 'rgba(100, 100, 100, 0.4)'

    # Grid lines
    y_grid = np.linspace(0, max_range, 11)
    x_max_at_y = max_range * np.tan(np.radians(max(np.abs(az) for az in az_fov_deg)))
    for y_val in y_grid:
        x_val = y_val * np.tan(np.radians(max(np.abs(az) for az in az_fov_deg)))
        traces.append(go.Scatter3d(x=[-x_val, x_val], y=[y_val, y_val], z=[0, 0], mode='lines', line=dict(color=grid_color, width=1), hoverinfo='none', showlegend=False, name=''))
    
    x_grid_lines = np.linspace(-x_max_at_y, x_max_at_y, 11)
    for x_val in x_grid_lines:
        traces.append(go.Scatter3d(x=[x_val, x_val], y=[0, max_range], z=[0, 0], mode='lines', line=dict(color=grid_color, width=1), hoverinfo='none', showlegend=False, name=''))

    # Radar cone surface with user-controlled opacity
    az_rad = np.radians(np.linspace(az_fov_deg[0], az_fov_deg[1], 50))
    el_rad = np.radians(np.linspace(-el_fov_deg / 2, el_fov_deg / 2, 25))
    x_outer = max_range * np.outer(np.cos(el_rad), np.sin(az_rad))
    y_outer = max_range * np.outer(np.cos(el_rad), np.cos(az_rad))
    z_outer = max_range * np.outer(np.sin(el_rad), np.ones_like(az_rad))
    
    traces.append(go.Surface(
        x=x_outer, y=y_outer, z=z_outer,
        colorscale=[[0, f'rgba(180, 200, 220, {cone_opacity})'], [1, f'rgba(180, 200, 220, {cone_opacity})']],
        showscale=False,
        hoverinfo='none',
        name='',
        showlegend=False
    ))
    return traces

def generate_true_4d_points(target_indices: Tuple[np.ndarray, np.ndarray], 
                            rd_map_complex_ch0: np.ndarray, 
                            rd_map_complex_ch1: np.ndarray, 
                            az_angle: float, scan_count: int) -> List[dict]:
    
    if not target_indices[0].size:
        return []

    # Constants for monopulse calculation
    c = 3e8
    wavelength = c / CONFIG['hardware']['OUTPUT_FREQ']
    element_spacing_el = 0.015

    points = []
    ch0_peaks = rd_map_complex_ch0[target_indices]
    ch1_peaks = rd_map_complex_ch1[target_indices]

    # Phase difference calculation
    delta_phase_rad = np.angle(ch0_peaks * np.conj(ch1_peaks))
    delta_phase_rad = np.unwrap(delta_phase_rad)

    # Elevation calculation
    arcsin_arg = np.clip((wavelength * delta_phase_rad) / (2 * np.pi * element_spacing_el), -1.0, 1.0)
    el_angles_rad = np.arcsin(arcsin_arg)

    # Get physical parameters
    ranges = range_m_axis[target_indices[1]]
    velocities = velocity_axis[target_indices[0]]
    
    # FIXED: Proper SNR calculation
    # SNR = 20*log10(signal_power / noise_floor)
    signal_power = np.abs(ch0_peaks)**2 + np.abs(ch1_peaks)**2
    
    # Estimate noise floor from surrounding cells
    noise_samples = []
    for doppler_idx, range_idx in zip(target_indices[0], target_indices[1]):
        # Sample noise from nearby cells
        d_start = max(0, doppler_idx - 3)
        d_end = min(rd_map_complex_ch0.shape[0], doppler_idx + 4)
        r_start = max(0, range_idx - 3) 
        r_end = min(rd_map_complex_ch0.shape[1], range_idx + 4)
        
        noise_region_ch0 = rd_map_complex_ch0[d_start:d_end, r_start:r_end]
        noise_region_ch1 = rd_map_complex_ch1[d_start:d_end, r_start:r_end]
        
        # Exclude the peak itself
        noise_region_ch0[doppler_idx - d_start, range_idx - r_start] = 0
        noise_region_ch1[doppler_idx - d_start, range_idx - r_start] = 0
        
        noise_power = np.mean(np.abs(noise_region_ch0)**2 + np.abs(noise_region_ch1)**2)
        noise_samples.append(noise_power)
    
    noise_floor = np.array(noise_samples)
    
    # Calculate SNR in dB
    snr_linear = signal_power / (noise_floor + 1e-12)
    snrs = 10 * np.log10(snr_linear + 1e-12)  # Proper SNR in dB
    
    # Create point cloud
    az_angle_rad = np.radians(az_angle)
    x = ranges * np.cos(el_angles_rad) * np.sin(az_angle_rad)
    y = ranges * np.cos(el_angles_rad) * np.cos(az_angle_rad)
    z = ranges * np.sin(el_angles_rad)
    
    current_time = time.time()
    for i in range(len(x)):
        points.append({
            'x': x[i], 'y': y[i], 'z': z[i],
            'velocity': velocities[i], 
            'snr': snrs[i],  # Now properly calculated
            'range': ranges[i], 'azimuth': az_angle,
            'timestamp': current_time, 'scan_count': scan_count
        })
        
    return points

def compute_chebyshev_weights(az_angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the required taper and phase weights for a given azimuth angle."""
    num_elements = CONFIG['hardware']['NUM_AZ_ELEMENTS']
    sidelobe_db = CONFIG['signal_processing']['SIDELOBE_DB']
    
    taper = chebwin(num_elements, at=sidelobe_db)
    
    rf_freq = CONFIG['hardware']['OUTPUT_FREQ']
    element_spacing = CONFIG['hardware']['ELEMENT_SPACING']
    c = 3e8
    
    element_indices = np.arange(num_elements)
    phase_rad = -2 * np.pi * rf_freq * element_spacing * np.sin(np.radians(az_angle)) / c * element_indices
    phase_deg = np.degrees(phase_rad)
    
    return taper, phase_deg

def cfar_2d_optimized(range_doppler_map_db, guard_cells=2, training_cells=8, false_alarm_rate=1e-6):
    """Proper 2D CFAR implementation"""
    
    threshold_map = np.full_like(range_doppler_map_db, -200.0)
    
    # Calculate adaptive threshold
    cfar_threshold = -np.log10(false_alarm_rate) * 10  # Convert to dB
    
    for r_idx in range(guard_cells + training_cells, range_doppler_map_db.shape[1] - guard_cells - training_cells):
        for d_idx in range(guard_cells + training_cells, range_doppler_map_db.shape[0] - guard_cells - training_cells):
            
            # Define training region (exclude guard cells)
            r_start = r_idx - guard_cells - training_cells
            r_end = r_idx + guard_cells + training_cells + 1
            d_start = d_idx - guard_cells - training_cells  
            d_end = d_idx + guard_cells + training_cells + 1
            
            # Training cells
            training_region = range_doppler_map_db[d_start:d_end, r_start:r_end]
            
            # Remove guard cells and cell under test
            guard_r_start = r_idx - guard_cells
            guard_r_end = r_idx + guard_cells + 1
            guard_d_start = d_idx - guard_cells
            guard_d_end = d_idx + guard_cells + 1
            
            training_region[guard_d_start-d_start:guard_d_end-d_start, 
                          guard_r_start-r_start:guard_r_end-r_start] = -200
            
            # Calculate noise level
            valid_training = training_region[training_region > -199]
            if len(valid_training) > 0:
                noise_level = np.mean(valid_training)
                threshold_map[d_idx, r_idx] = noise_level + cfar_threshold
    
    return threshold_map

def process_live_data_with_micro_doppler(iq_ch0: np.ndarray, iq_ch1: np.ndarray, scan_count: int, az_angle: float):
    """
    ENHANCED: Add micro-Doppler drone detection to existing processing.
    Integrates with your existing CN0566 radar processing pipeline.
    """
    try:
        with RADAR_DATA_LOCK:
            active_profile_name = RADAR_DATA.get('active_profile_name', 'indoor')
        profile = INDOOR_PROFILE if active_profile_name == 'indoor' else OUTDOOR_PROFILE
        num_chirps, samples_per_chirp = CONFIG['hardware']['NUM_CHIRPS'], CONFIG['hardware']['SAMPLES_PER_CHIRP']
        
        if iq_ch0.size < num_chirps * samples_per_chirp: 
            return

        # === ENHANCED FMCW PROCESSING (Based on Jon Kraft's CN0566 implementations) ===
        def process_channel_enhanced_fmcw(iq_data, remove_dc=True):
            """
            Enhanced FMCW processing incorporating Jon Kraft's proven algorithms
            from FMCW_RADAR_Waterfall_ChirpSync.py and related implementations
            """
            num_chirps = CONFIG['hardware']['NUM_CHIRPS']
            samples_per_chirp = CONFIG['hardware']['SAMPLES_PER_CHIRP']
            
            if iq_data.size < num_chirps * samples_per_chirp:
                logger.warning(f"Insufficient IQ data: {iq_data.size} < {num_chirps * samples_per_chirp}")
                return np.zeros((num_chirps, len(range_m_axis)), dtype=np.complex128)
            
            try:
                # Reshape and ensure proper data type (Jon Kraft's approach)
                matrix = iq_data[:num_chirps * samples_per_chirp].reshape((num_chirps, samples_per_chirp))
                matrix = matrix.astype(np.complex128)
                
                # Enhanced DC removal (per Jon Kraft's FMCW implementations)
                if remove_dc:
                    # Method 1: Per-chirp DC removal (most effective for FMCW)
                    for chirp_idx in range(num_chirps):
                        dc_i = np.mean(np.real(matrix[chirp_idx, :]))
                        dc_q = np.mean(np.imag(matrix[chirp_idx, :]))
                        matrix[chirp_idx, :] -= (dc_i + 1j * dc_q)
                    
                    # Method 2: Remove residual DC across all chirps
                    mean_dc = np.mean(matrix)
                    matrix -= mean_dc
                
                # Enhanced windowing (Jon Kraft's proven approach)
                # Use Blackman-Harris window for better sidelobe suppression
                range_window = np.blackman(samples_per_chirp).astype(np.complex128)
                matrix_windowed = matrix * range_window[np.newaxis, :]
                
                # Range FFT with proper scaling (Jon Kraft's method)
                range_fft = np.fft.fft(matrix_windowed, axis=1)
                # Take only positive frequencies for FMCW
                range_fft = range_fft[:, :samples_per_chirp//2]
                
                # Proper normalization for dBFS calculations
                range_fft = range_fft / samples_per_chirp
                
                # Enhanced Doppler processing with proper windowing
                doppler_window = np.blackman(num_chirps).astype(np.complex128)
                range_fft_windowed = range_fft * doppler_window[:, np.newaxis]
                
                # Doppler FFT with fftshift for proper velocity mapping
                rd_map_complex = np.fft.fftshift(np.fft.fft(range_fft_windowed, axis=0), axes=0)
                rd_map_complex = rd_map_complex / num_chirps  # Proper normalization
                
                # Phase calibration correction (important for coherent processing)
                # Apply small phase correction to compensate for timing jitter
                if hasattr(process_channel_enhanced_fmcw, 'phase_reference'):
                    # Use stored phase reference for correction
                    phase_correction = np.angle(np.mean(rd_map_complex[num_chirps//2-2:num_chirps//2+2, 1:5]))
                    rd_map_complex *= np.exp(-1j * phase_correction)
                else:
                    # Initialize phase reference
                    process_channel_enhanced_fmcw.phase_reference = np.angle(np.mean(rd_map_complex[num_chirps//2-2:num_chirps//2+2, 1:5]))
                
                # Ensure output matches expected dimensions
                output = rd_map_complex[:, :len(range_m_axis)]
                
                return output
                
            except Exception as e:
                logger.error(f"Enhanced FMCW processing failed: {e}")
                # Fallback to basic processing
                return process_channel_basic_fallback(iq_data, remove_dc)
        
        def process_channel_basic_fallback(iq_data, remove_dc=True):
            """Fallback processing method"""
            num_chirps = CONFIG['hardware']['NUM_CHIRPS']
            samples_per_chirp = CONFIG['hardware']['SAMPLES_PER_CHIRP']
            
            try:
                matrix = iq_data[:num_chirps * samples_per_chirp].reshape((num_chirps, samples_per_chirp))
                matrix = matrix.astype(np.complex128)
                
                if remove_dc:
                    matrix = matrix - np.mean(matrix)
                
                # Basic processing
                range_fft = np.fft.fft(matrix, axis=1)[:, :samples_per_chirp//2]
                rd_map_complex = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
                
                return rd_map_complex[:, :len(range_m_axis)]
                
            except Exception as e:
                logger.error(f"Fallback processing failed: {e}")
                return np.zeros((num_chirps, len(range_m_axis)), dtype=np.complex128)
        
        # Use enhanced processing as primary method
        process_channel = process_channel_enhanced_fmcw

        rd_map_complex_ch0 = process_channel(iq_ch0)
        rd_map_complex_ch1 = process_channel(iq_ch1)
        
        # === ENHANCED: MTI PROCESSING FOR MOVING TARGET INDICATION ===
        if not hasattr(process_live_data_with_micro_doppler, 'mti_history_ch0'):
            process_live_data_with_micro_doppler.mti_history_ch0 = deque(maxlen=5)
            process_live_data_with_micro_doppler.mti_history_ch1 = deque(maxlen=5)
        
        # Store raw data for MTI processing
        process_live_data_with_micro_doppler.mti_history_ch0.append(rd_map_complex_ch0)
        process_live_data_with_micro_doppler.mti_history_ch1.append(rd_map_complex_ch1)
        
        # Apply enhanced MTI processing to both channels
        if len(process_live_data_with_micro_doppler.mti_history_ch0) >= 3:
            rd_map_complex_ch0_mti = enhanced_mti_processing(
                list(process_live_data_with_micro_doppler.mti_history_ch0), mti_mode='3pulse'
            )
            rd_map_complex_ch1_mti = enhanced_mti_processing(
                list(process_live_data_with_micro_doppler.mti_history_ch1), mti_mode='3pulse'
            )
        else:
            # Use original data if insufficient history
            rd_map_complex_ch0_mti = rd_map_complex_ch0
            rd_map_complex_ch1_mti = rd_map_complex_ch1
        
        # FIXED: Proper combination and dB conversion with MTI data
        combined_rd_map_complex = (rd_map_complex_ch0_mti + rd_map_complex_ch1_mti) / 2
        combined_rd_map_db = 20 * np.log10(np.abs(combined_rd_map_complex) + 1e-12)
        
        # FIXED: Clip unrealistic values
        combined_rd_map_db = np.clip(combined_rd_map_db, -120, 20)  # Reasonable dB range

        # === ENHANCED: MICRO-DOPPLER HISTORY + MULTIPATH SUPPRESSION ===
        if not hasattr(process_live_data_with_micro_doppler, 'rd_history'):
            process_live_data_with_micro_doppler.rd_history = deque(maxlen=30)  # 3 seconds at 10 Hz
        if not hasattr(process_live_data_with_micro_doppler, 'rd_complex_history'):
            process_live_data_with_micro_doppler.rd_complex_history = deque(maxlen=5)  # For multipath suppression
        
        # Store complex data for multipath suppression
        process_live_data_with_micro_doppler.rd_complex_history.append(combined_rd_map_complex)
        
        # --- APPLY COHERENT MULTIPATH SUPPRESSION ---
        suppressed_rd_map_complex = coherent_multipath_suppression(
            list(process_live_data_with_micro_doppler.rd_complex_history)
        )
        
        # Use the suppressed map for all subsequent processing
        combined_rd_map_db = 20 * np.log10(np.abs(suppressed_rd_map_complex) + 1e-12)

        # This forces the zero-velocity bin to be clean, fixing the "red static object" issue.
        center_doppler_bin = combined_rd_map_db.shape[0] // 2
        combined_rd_map_db[center_doppler_bin - 1 : center_doppler_bin + 2, :] = -120
        
        # Store for micro-Doppler analysis (use original complex data)
        process_live_data_with_micro_doppler.rd_history.append(combined_rd_map_complex)

        # === ENHANCED: ADVANCED CFAR DETECTION ===
        if CFAR_AVAILABLE:
            # Use advanced CFAR detection with multiple methods
            cfar_threshold_map, detected_targets = advanced_cfar_detection(
                np.abs(suppressed_rd_map_complex),
                num_guard_cells=2, num_ref_cells=8,
                cfar_bias=profile.get('MIN_PEAK_HEIGHT_DB', -60),
                cfar_method='average'
            )
            target_mask = detected_targets > 0
        else:
            # Fallback to existing threshold detection
            threshold_map = np.full_like(combined_rd_map_db, profile["MIN_PEAK_HEIGHT_DB"])
            for r_idx in range(combined_rd_map_db.shape[1]):
                current_range = range_m_axis[r_idx] if r_idx < len(range_m_axis) else 0
                if current_range > 4.0:  # Beyond room size, increase threshold
                    threshold_map[:, r_idx] += 10 + (current_range - 4.0) * 5
            target_mask = combined_rd_map_db > threshold_map

        # Find peaks above adaptive threshold
        target_indices = np.where(target_mask)
        
        # === ENHANCED: MICRO-DOPPLER DRONE DETECTION ===
        drone_signatures = []
        if len(process_live_data_with_micro_doppler.rd_history) >= 20:
            drone_signatures = enhanced_drone_micro_doppler_classification(
                list(process_live_data_with_micro_doppler.rd_history), 
                target_indices,
                frame_rate=10  # Your system runs at ~10 Hz
            )
            
            # Log drone detections with enhanced information
            if drone_signatures:
                logger.info(f"🚁 ENHANCED DRONE DETECTION: {len(drone_signatures)} signatures at azimuth {az_angle:.1f}°")
                for drone_sig in drone_signatures:
                    logger.info(f"  - {drone_sig['type']} at range {drone_sig.get('position', [0,0,0])[1]:.1f}m")
                    logger.info(f"    Confidence: {drone_sig['confidence']:.2f}, Signal: {drone_sig['signal_strength']:.2e}")
                    logger.info(f"    Scores - Blade: {drone_sig['blade_flash_score']:.2f}, "
                               f"Harmonic: {drone_sig['harmonic_score']:.2f}, "
                               f"Spectral: {drone_sig['spectral_spread']:.2f}")
        
        # === EXISTING PEAK DETECTION (maintained for compatibility) ===


        # === EXISTING POINT GENERATION ===
        new_points_for_angle = generate_true_4d_points(target_indices, rd_map_complex_ch0, rd_map_complex_ch1, az_angle, scan_count)
        
        # === NEW: ENHANCE POINTS WITH MICRO-DOPPLER CLASSIFICATION ===
        for drone_sig in drone_signatures:
            for point in new_points_for_angle:
                # Match drone signature to radar points
                if (abs(point['range'] - drone_sig['range']) < 0.5 and 
                    abs(point['velocity'] - drone_sig['velocity']) < 0.3):
                    
                    # Tag as confirmed drone with micro-Doppler evidence
                    point['micro_doppler_type'] = drone_sig['type']
                    point['drone_class'] = drone_sig['drone_class']
                    point['blade_frequency'] = drone_sig['blade_frequency']
                    point['threat_level'] = drone_sig['threat_level']
                    point['drone_confidence'] = drone_sig['confidence']
                    point['propeller_count'] = drone_sig['propeller_count']
                    
                    # Override SNR for confirmed drones (make them more visible)
                    point['snr'] = max(point['snr'], -50.0)  # Boost visibility
                    break

        # === EXISTING FILTERING AND STORAGE ===
        # Filter out points beyond physical room boundaries for indoor mode
        if active_profile_name == 'indoor':
            filtered_points = []
            for point in new_points_for_angle:
                if (abs(point['x']) <= 2.0 and  # Room width constraint
                    point['y'] <= 4.0 and point['y'] >= 0 and  # Room length constraint
                    abs(point['z']) <= 1.25):  # Room height constraint
                    filtered_points.append(point)
            new_points_for_angle = filtered_points
        
        # === EXISTING BEAMPATTERN AND DATA STORAGE ===
        taper_weights, phase_weights_deg = compute_chebyshev_weights(az_angle)
        beampattern_for_plot = compute_beampattern_plot_data(taper_weights, phase_weights_deg)
        
        with RADAR_DATA_LOCK:
            RADAR_DATA['persistent_points'][az_angle] = deque(new_points_for_angle, maxlen=200)
            RADAR_DATA['range_doppler_map'] = combined_rd_map_db
            RADAR_DATA['waterfall'].append(np.max(combined_rd_map_db, axis=0))
            RADAR_DATA['beampattern'] = beampattern_for_plot
            RADAR_DATA['current_az'] = az_angle
            RADAR_DATA['scan_progress'] = (list(azimuth_angles_scan).index(az_angle) + 1) / len(azimuth_angles_scan)
            RADAR_DATA['data_is_ready'] = True
            RADAR_DATA['last_update'] = time.time()
            
            # NEW: Store drone detection statistics
            RADAR_DATA['drone_detections'] = len(drone_signatures)
            RADAR_DATA['last_drone_detection'] = time.time() if drone_signatures else RADAR_DATA.get('last_drone_detection', 0)
            
            if abs(az_angle - azimuth_angles_scan[0]) < 0.1: 
                RADAR_DATA['scan_count'] += 1
                
    except Exception as e:
        logger.error(f"Error in process_live_data_with_micro_doppler: {e}\n{traceback.format_exc()}")


def get_laptop_ip_addresses():
    """SIMPLIFIED: Return simple IP list without netifaces dependency"""
    # Simple approach - return common IPs
    return ["192.168.0.6", "127.0.0.1", "0.0.0.0"]
    """Get all possible IP addresses for this laptop"""
    addresses = []
    
    try:
        # Method 1: Try to get IP via connection test
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        addresses.append(primary_ip)
    except:
        pass
    
    # Method 2: Common laptop IPs based on network type  
    # NOTE: These are updated based on actual WiFi network discovery
    common_ips = [
        "192.168.0.6",      # WiFi (actual current setup - laptop IP)
        "192.168.137.1",    # Hotspot mode
        "192.168.100.1",    # Ethernet
        "192.168.1.100",    # Alternative router setup
        "10.0.0.1",         # Alternative
    ]
    
    for ip in common_ips:
        if ip not in addresses:
            addresses.append(ip)
            
    return addresses

def receiver_thread():
    """SIMPLIFIED: Receiver based on working dashboard_host.py approach - FUNCTIONAL FIRST"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    
    # SIMPLE BINDING like working version - just bind to all interfaces
    try:
        s.bind(('', LAPTOP_SERVER_PORT))  # Bind to all interfaces like working version
        logger.info(f"✅ UDP server listening on ALL INTERFACES port {LAPTOP_SERVER_PORT}")
        logger.info("📡 Ready to receive data from Pi transmitter...")
    except Exception as e:
        logger.error(f"❌ Failed to bind to port {LAPTOP_SERVER_PORT}: {e}")
        return

    frame_assembler = {}
    total_frames_attempted = 0
    successful_frames = 0
    dropped_frames = 0

    try:
        s.settimeout(0.1)  # Set timeout for socket operations
        while True:
            try:
                data, addr = s.recvfrom(65536)
                
                try:
                    data_packet = pickle.loads(data)
                except Exception as e:
                    logger.error(f"❌ Pickle deserialization failed: {e}")
                    continue
                
                # Handle heartbeat packets like working version
                if isinstance(data_packet, dict) and 'command' in data_packet:
                    command = data_packet.get('command')
                    
                    if command == 'HEALTH_STATUS':
                        with RADAR_DATA_LOCK:
                            RADAR_DATA['last_heartbeat_time'] = time.time()
                        logger.info(f"💓 Heartbeat from {addr}")
                    elif command == 'SET_PROFILE':
                        profile_name = data_packet.get('profile', 'indoor')
                        with RADAR_DATA_LOCK:
                            RADAR_DATA['active_profile_name'] = profile_name
                        logger.info(f"🔧 Profile switched to: {profile_name}")
                    continue

                # Process I/Q data packets - SIMPLIFIED like working version
                if isinstance(data_packet, dict) and 'iq_data_chunk' in data_packet:
                    frame_num = data_packet.get('frame', 0)
                    chunk_index = data_packet.get('chunk_index', 0)
                    total_chunks = data_packet.get('total_chunks', 1)
                    az_angle = data_packet.get('az_angle', 0.0)
                    
                    # Log first few packets for debugging
                    if total_frames_attempted < 10:
                        logger.info(f"📦 Received chunk {chunk_index}/{total_chunks} for frame {frame_num} "
                                   f"at {az_angle}° from {addr}")
                    
                    # SIMPLE frame assembly like working version
                    if frame_num not in frame_assembler:
                        frame_assembler[frame_num] = {
                            'chunks': {},
                            'az_angle': az_angle,
                            'total_chunks': total_chunks,
                            'timestamp': time.time()
                        }
                    
                    frame_assembler[frame_num]['chunks'][chunk_index] = data_packet['iq_data_chunk']
                    
                    # Check if frame is complete
                    if len(frame_assembler[frame_num]['chunks']) == total_chunks:
                        # Reassemble data
                        iq_data_chunks = []
                        for i in range(total_chunks):
                            if i in frame_assembler[frame_num]['chunks']:
                                iq_data_chunks.append(frame_assembler[frame_num]['chunks'][i])
                        
                        if len(iq_data_chunks) == total_chunks:
                            # Combine chunks back into full arrays
                            ch0_combined = np.concatenate([chunk[0] for chunk in iq_data_chunks])
                            ch1_combined = np.concatenate([chunk[1] for chunk in iq_data_chunks])
                            
                            # Process the data - KEEP ROBUST PROCESSING but simpler logging
                            try:
                                process_live_data_with_micro_doppler(
                                    ch0_combined, ch1_combined, 
                                    frame_num, frame_assembler[frame_num]['az_angle']
                                )
                                successful_frames += 1
                                
                                # Reduced logging for better performance
                                if successful_frames % 10 == 0:  # Every 10 frames
                                    logger.info(f"✅ Processed {successful_frames} frames successfully")
                                    
                            except Exception as e:
                                logger.error(f"❌ Processing failed for frame {frame_num}: {e}")
                        
                        # Clean up completed frame
                        del frame_assembler[frame_num]
                        total_frames_attempted += 1
                
                # Clean up old incomplete frames
                current_time = time.time()
                frames_to_remove = []
                for frame_num, frame_data in frame_assembler.items():
                    if current_time - frame_data['timestamp'] > 5.0:  # 5 second timeout
                        frames_to_remove.append(frame_num)
                        dropped_frames += 1
                
                for frame_num in frames_to_remove:
                    del frame_assembler[frame_num]
                    if dropped_frames % 10 == 0:  # Reduced logging
                        logger.warning(f"⚠️ Dropped {dropped_frames} incomplete frames")

                # Periodic stats logging
                if total_frames_attempted > 0 and total_frames_attempted % 100 == 0:  # Every 100 frames
                    success_rate = (successful_frames / total_frames_attempted) * 100
                    logger.info(f"📊 Stats: {successful_frames}/{total_frames_attempted} successful "
                               f"({success_rate:.1f}%), {dropped_frames} dropped")

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"❌ Receiver error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"❌ Critical receiver error: {e}")
    finally:
        s.close()
        logger.info("🧹 Receiver thread stopped")

def monitor_network_performance():
    """Monitor network performance - simplified to use configured PI_IP."""
    import subprocess
    import re
    
    try:
        # Use configured PI_IP instead of dynamic detection
        try:
            # Use -n 1 for Windows ping syntax
            result = subprocess.run(['ping', '-n', '1', PI_IP],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Extract ping time (Windows format)
                ping_match = re.search(r'time[<=](\d+\.?\d*)ms', result.stdout)
                if ping_match:
                    ping_time = float(ping_match.group(1))
                    if ping_time > 10:  # Alert if ping > 10ms
                        logger.warning(f"High network latency to Pi ({PI_IP}): {ping_time}ms")
                    else:
                        logger.info(f"Pi connectivity OK ({PI_IP}): {ping_time}ms")
                    return ping_time
                else:
                    logger.info(f"Pi reachable at {PI_IP} (ping time not parsed)")
                    return 1.0  # Assume good latency
        except Exception as e:
            logger.debug(f"Ping to {PI_IP} failed: {e}")
                
        logger.warning(f"Pi not detected at configured IP: {PI_IP}")
    except Exception as e:
        logger.debug(f"Network monitoring failed: {e}")
    return None

# Add this to your performance monitoring
network_latency = monitor_network_performance()


# ==== SECTION: MAVLink Exporter Thread ====
def mavlink_exporter_thread():
    """
    NEW FUNCTION (Runs in a separate thread on the Laptop)
    Connects to the drone as a GCS and sends OBSTACLE_DISTANCE MAVLink messages
    based on the confirmed tracks from our radar system.
    """
    from pymavlink import mavutil  # <-- Import mavutil here

    # --- MAVLINK CONNECTION ---
    # This should be the IP address of your drone on the network.
    # The port 14550 is the standard for MAVLink telemetry.
    DRONE_IP = "192.168.0.105" #<-- IMPORTANT: Change this to your drone's IP
    CONNECTION_STRING = f"udp:{DRONE_IP}:14550"

    try:
        # Establish a connection that listens on 14551 and sends to 14550
        mav_conn = mavutil.mavlink_connection(CONNECTION_STRING)
        mav_conn.wait_heartbeat()
        logger.info(f"MAVLink Exporter connected to vehicle on {DRONE_IP}:14550!")
    except Exception as e:
        logger.error(f"MAVLink Exporter failed to connect: {e}")
        return

    while True:
        try:
            with RADAR_DATA_LOCK:
                # Make a safe copy of the tracked objects to work with
                current_tracks = list(TRACKED_OBJECTS.values())

            # Find the closest confirmed object directly in front of the drone
            closest_range = float('inf')
            forward_threat_found = False
            for track in current_tracks:
                if not track.get('is_confirmed', False):
                    continue

                # Consider a "forward cone" of +/- 20 degrees azimuth
                # We can get the azimuth from the object's x/y position
                azimuth_deg = np.degrees(np.arctan2(track['centroid'][0], track['centroid'][1]))
                if abs(azimuth_deg) < 20.0:
                    dist = np.linalg.norm(track['centroid'])
                    if dist < closest_range:
                        closest_range = dist
                        forward_threat_found = True
            
            # ArduCopter's OBSTACLE_DISTANCE message has 72 "slots", each covering a 5-degree arc.
            # 0 is forward, 1 is 5deg right, 71 is 5deg left.
            distances_cm = [65535] * 72 # Initialize all distances to "unknown" (max range)
            min_dist_cm = 65535
            
            if forward_threat_found:
                min_dist_cm = int(closest_range * 100) # Convert to cm
                # For simplicity, we'll place the obstacle directly in the forward-facing slot.
                distances_cm[0] = min_dist_cm

            # Send the message to the autopilot
            mav_conn.mav.obstacle_distance_send(
                0, # time_usec
                12, # sensor_type: MAV_SENSOR_TYPE_RADAR
                distances_cm, # distances array in cm
                0, # increment_f (not used if all distances are sent)
                100, # min_distance in cm
                int(CONFIG['signal_processing']['MAX_RANGE_DISPLAY'] * 100), # max_distance in cm
                -1, # increment
                -1, # angle_offset
                12 # MAV_FRAME, frame of reference
            )

            # Send updates at 10Hz
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"MAVLink Exporter thread error: {e}")
            time.sleep(1) # Wait before retrying

# ==== SECTION: Surface Reconstruction Functions ====
# === HELPER FUNCTIONS FOR KALMAN VISUALIZATION ===
def create_kalman_bounding_box_trace(position, box_size, color):
    """Create a simple bounding box around a Kalman tracker position."""
    x, y, z = position
    
    # Define box corners
    corners = np.array([
        [x-box_size, y-box_size, z-box_size], [x+box_size, y-box_size, z-box_size],
        [x+box_size, y+box_size, z-box_size], [x-box_size, y+box_size, z-box_size],
        [x-box_size, y-box_size, z+box_size], [x+box_size, y-box_size, z+box_size],
        [x+box_size, y+box_size, z+box_size], [x-box_size, y+box_size, z+box_size]
    ])
    
    # Define edges
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # Bottom face
        [4,5], [5,6], [6,7], [7,4],  # Top face
        [0,4], [1,5], [2,6], [3,7]   # Vertical edges
    ]
    
    x_lines, y_lines, z_lines = [], [], []
    for edge in edges:
        p1, p2 = corners[edge[0]], corners[edge[1]]
        x_lines.extend([p1[0], p2[0], None])
        y_lines.extend([p1[1], p2[1], None])
        z_lines.extend([p1[2], p2[2], None])
    
    return go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color=color, width=4),
        showlegend=False,
        hoverinfo='none',
        name=''
    )

def create_velocity_vector_trace(position, velocity, color):
    """Create a velocity vector visualization."""
    if np.linalg.norm(velocity) < 0.1:  # Don't show very small velocities
        return None
    
    x, y, z = position
    vx, vy, vz = velocity * 2  # Scale for visibility
    
    return go.Scatter3d(
        x=[x, x + vx], y=[y, y + vy], z=[z, z + vz],
        mode='lines+markers',
        line=dict(color=color, width=6),
        marker=dict(size=[4, 8], color=color),
        showlegend=False,
        hovertemplate=f'Velocity: {np.linalg.norm(velocity):.2f} m/s<extra></extra>',
        name=''
    )

def create_voxel_grid_with_semantics(df_points, grid_resolution=0.25, voxel_size=6, semantic_objects=None):
    """
    FIXED: Proper semantic voxel colors and classification
    """
    if df_points.empty:
        logger.info("create_voxel_grid_with_semantics: No points provided")
        return None

    logger.info(f"create_voxel_grid_with_semantics: Processing {len(df_points)} points")

    # Create voxel grid coordinates
    df_points_copy = df_points.copy()
    df_points_copy['vx'] = np.floor(df_points_copy['x'] / grid_resolution) * grid_resolution
    df_points_copy['vy'] = np.floor(df_points_copy['y'] / grid_resolution) * grid_resolution
    df_points_copy['vz'] = np.floor(df_points_copy['z'] / grid_resolution) * grid_resolution

    # Aggregate points within each voxel, keeping the max SNR
    try:
        voxel_df = df_points_copy.loc[df_points_copy.groupby(['vx', 'vy', 'vz'])['snr'].idxmax()]
    except Exception as e:
        logger.error(f"Voxel aggregation failed: {e}")
        return None

    # Initialize with default classification
    voxel_df = voxel_df.copy()
    voxel_df['semantic_type'] = 'Unknown'
    voxel_df['confidence'] = 0.0
    voxel_df['threat_level'] = 'NONE'

    # ENHANCED: Check for micro-Doppler signatures first
    if 'micro_doppler_type' in df_points_copy.columns:
        drone_mask = df_points_copy['micro_doppler_type'] == 'DRONE_CONFIRMED'
        if drone_mask.any():
            # Find voxels that contain drone points
            drone_points = df_points_copy[drone_mask]
            for _, drone_point in drone_points.iterrows():
                # Find the voxel that contains this drone point
                drone_voxel_mask = (
                    (voxel_df['vx'] == np.floor(drone_point['x'] / grid_resolution) * grid_resolution) &
                    (voxel_df['vy'] == np.floor(drone_point['y'] / grid_resolution) * grid_resolution) &
                    (voxel_df['vz'] == np.floor(drone_point['z'] / grid_resolution) * grid_resolution)
                )
                
                if drone_voxel_mask.any():
                    voxel_df.loc[drone_voxel_mask, 'semantic_type'] = 'DRONE_THREAT'
                    voxel_df.loc[drone_voxel_mask, 'confidence'] = 0.95
                    voxel_df.loc[drone_voxel_mask, 'threat_level'] = 'HIGH'

    # Add classification for moving objects
    moving_mask = (voxel_df['velocity'] > 1.0) & (voxel_df['velocity'] < 2.0)
    if moving_mask.any():
        voxel_df.loc[moving_mask, 'semantic_type'] = 'MOVING_OBJECT'
        voxel_df.loc[moving_mask, 'confidence'] = 0.8
        voxel_df.loc[moving_mask, 'threat_level'] = 'LOW'

    # Add classification for static objects
    static_mask = (voxel_df['velocity'] <= 0.1)
    if static_mask.any():
        voxel_df.loc[static_mask, 'semantic_type'] = 'STATIC_OBJECT'
        voxel_df.loc[static_mask, 'confidence'] = 0.7
        voxel_df.loc[static_mask, 'threat_level'] = 'NONE'

    # Apply other semantic classifications if we have verified objects
    if semantic_objects and len(semantic_objects) > 0:
        logger.info(f"Applying {len(semantic_objects)} semantic classifications")
        
        for semantic_obj in semantic_objects:
            # Only use high-confidence semantic objects
            if semantic_obj['confidence'] < 0.85:  # Increased threshold
                continue
                
            obj_pos = semantic_obj['position']
            obj_vx = np.floor(obj_pos[0] / grid_resolution) * grid_resolution
            obj_vy = np.floor(obj_pos[1] / grid_resolution) * grid_resolution
            obj_vz = np.floor(obj_pos[2] / grid_resolution) * grid_resolution
            
            # Find voxels within semantic object influence
            influence_radius = grid_resolution * 1.2  # Smaller influence
            nearby_mask = (
                (np.abs(voxel_df['vx'] - obj_vx) <= influence_radius) &
                (np.abs(voxel_df['vy'] - obj_vy) <= influence_radius) &
                (np.abs(voxel_df['vz'] - obj_vz) <= influence_radius) &
                (voxel_df['semantic_type'] == 'Unknown')  # Don't override existing classifications
            )
            
            if nearby_mask.any():
                # Update voxels with semantic information
                voxel_df.loc[nearby_mask, 'semantic_type'] = semantic_obj['type']
                voxel_df.loc[nearby_mask, 'confidence'] = semantic_obj['confidence']
                voxel_df.loc[nearby_mask, 'threat_level'] = semantic_obj['threat_level']
    
    # FIXED: Correct color mapping
    color_map = {
        'DRONE_THREAT': '#FF0000',      # Red - drones
        'STRUCTURE': '#00FFFF',         # Cyan - walls/barriers  
        'MOVING_OBJECT': '#00FF00',     # Green - moving objects
        'STATIC_OBJECT': '#FFA500',     # Orange - static objects/furniture
        'OBJECT': '#FFA500',            # Orange - furniture (legacy)
        'DETECTION': '#00FFFF',         # Cyan - unclassified
        'Unknown': '#8A2BE2'            # Blue-Violet - default
    }
    
    # Assign colors based on semantic type
    voxel_colors = [color_map.get(sem_type, color_map['Unknown']) for sem_type in voxel_df['semantic_type']]
    
    # Size based on confidence and SNR
    base_size = voxel_size
    confidence_scale = voxel_df['confidence'] * 0.5 + 0.5  # 0.5 to 1.0 scale
    snr_scale = np.clip((voxel_df['snr'] + 100) / 50, 0.3, 1.5)  # SNR scaling
    voxel_sizes = base_size * confidence_scale * snr_scale * 1.5  # Scale sizes for visibility

    logger.info(f"create_voxel_grid_with_semantics: Created {len(voxel_df)} voxels with colors: {set(voxel_df['semantic_type'])}")

    # Create the voxel trace
    voxel_trace = go.Scatter3d(
        x=voxel_df['vx'],
        y=voxel_df['vy'],
        z=voxel_df['vz'],
        mode='markers',
        marker=dict(
            symbol='square',
            size=voxel_sizes,
            color=voxel_colors,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        hovertemplate=(
            'Type: %{customdata[0]}<br>'
            'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>'
            'SNR: %{customdata[1]:.1f}dB<br>'
            'Confidence: %{customdata[2]:.0%}<br>'
            'Threat Level: %{customdata[3]}<extra></extra>'
        ),
        customdata=np.column_stack([
            voxel_df['semantic_type'],
            voxel_df['snr'],
            voxel_df['confidence'],
            voxel_df['threat_level']
        ]),
        name='Semantic Voxels',
        showlegend=False
    )
    
    return voxel_trace


# Simplify performance mode logic:
def get_performance_mode():
    """Simplified: just auto-detect between full and minimal"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.01)
        memory_info = psutil.virtual_memory()
        
        if cpu_percent > 80 or memory_info.percent > 85:
            return 'minimal'
        else:
            return 'full'
    except:
        return 'full'

def memory_efficient_point_sampling(df, max_points=2000, memory_budget_mb=50):
    """
    OPTIMIZED: Vectorized sampling with memory constraints for i5-1230U.
    Uses Intel-specific optimizations.
    """
    if len(df) <= max_points:
        return df
    
    try:
        # Vectorized memory check
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if memory_usage > memory_budget_mb:
            reduction_factor = memory_budget_mb / memory_usage
            max_points = int(max_points * reduction_factor)
        
        # Intel-optimized sorting using numpy
        snr_weights = df['snr'].values
        time_weights = df['timestamp'].values
        
        # Vectorized weighted sampling
        combined_weights = (snr_weights + 100) / 60 + (time_weights - time_weights.min()) / (time_weights.max() - time_weights.min() + 1e-8)
        
        # Use numpy's argpartition for O(n) selection instead of O(n log n) sorting
        top_indices = np.argpartition(combined_weights, -max_points)[-max_points:]
        
        return df.iloc[top_indices]
    except:
        return df.sample(n=min(max_points, len(df)))

def adaptive_dense_cloud_generation(df_points, performance_mode='balanced',
                                    semantic_objects=None, voxel_resolution=0.4, voxel_size=6): # Add parameter
    # ...
    if performance_mode == 'minimal':
        res = max(voxel_resolution, 0.5)
        voxel_trace = create_voxel_grid_with_semantics(df_points, grid_resolution=res, voxel_size=voxel_size, semantic_objects=semantic_objects) # Pass it
        return [voxel_trace] if voxel_trace else []
    else: # 'balanced' or 'full'
        voxel_trace = create_voxel_grid_with_semantics(df_points, grid_resolution=voxel_resolution, voxel_size=voxel_size, semantic_objects=semantic_objects) # Pass it
        return [voxel_trace] if voxel_trace else []

def optimize_kalman_trackers(performance_mode='balanced'):
    """
    Optimize Kalman tracker processing based on performance mode.
    """
    global KALMAN_TRACKERS
    
    if performance_mode == 'minimal':
        # Limit number of active trackers
        max_trackers = 5
        if len(KALMAN_TRACKERS) > max_trackers:
            # Keep only the most recent trackers
            sorted_trackers = sorted(KALMAN_TRACKERS.items(), 
                                   key=lambda x: x[1].last_seen, reverse=True)
            KALMAN_TRACKERS = dict(sorted_trackers[:max_trackers])
    
    elif performance_mode == 'balanced':
        # Limit tracker history
        for tracker in KALMAN_TRACKERS.values():
            if len(tracker.history) > CONFIG['performance']['KALMAN_HISTORY_LIMIT']:
                tracker.history = tracker.history[-CONFIG['performance']['KALMAN_HISTORY_LIMIT']:]

def optimize_intel_cpu():
    """
    Intel i5-1230U specific optimizations.
    """
    import os
    
    # Set Intel MKL threading for optimal performance
    os.environ['MKL_NUM_THREADS'] = '4'  # P-cores only for heavy computation
    os.environ['OMP_NUM_THREADS'] = '10'  # All cores for parallel tasks
    os.environ['NUMEXPR_MAX_THREADS'] = '10'
    
    # Intel-specific optimizations
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    # Enable Intel's fast math
    try:
        import numpy as np
        np.seterr(all='ignore')  # Ignore floating point errors for speed
    except:
        pass

def performance_profiler(func):
    """Decorator to profile function performance."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"{func.__name__}: {(end_time - start_time)*1000:.2f}ms")
        return result
    return wrapper

# Apply to critical functions
adaptive_dense_cloud_generation = performance_profiler(adaptive_dense_cloud_generation)

# ==== SECTION: Enhanced Micro-Doppler Drone Detection ====
def enhanced_drone_micro_doppler_classification(rd_map_complex_history, target_indices, frame_rate=10):
    """
    ENHANCED: Advanced micro-Doppler drone classification with multiple detection methods
    Combines spectral analysis, rotor blade flash detection, and harmonic analysis
    """
    if len(rd_map_complex_history) < 25:
        return []

    drone_detections = []
    
    # Enhanced target filtering with adaptive thresholds
    strong_targets = []
    for doppler_idx, range_idx in zip(target_indices[0], target_indices[1]):
        if (doppler_idx >= len(rd_map_complex_history[0]) or 
            range_idx >= len(rd_map_complex_history[0][0])):
            continue
            
        latest_frame = rd_map_complex_history[-1]
        signal_strength = np.abs(latest_frame[doppler_idx, range_idx])
        
        # Adaptive threshold based on background noise
        background_noise = np.median(np.abs(latest_frame))
        adaptive_threshold = background_noise * 25  # 25x background
        
        if signal_strength > adaptive_threshold:
            strong_targets.append((doppler_idx, range_idx, signal_strength))
    
    # Sort by signal strength for prioritized analysis
    strong_targets.sort(key=lambda x: x[2], reverse=True)
    
    for doppler_idx, range_idx, signal_strength in strong_targets[:10]:  # Analyze top 10 targets
        try:
            # Extract time series for this target
            target_history = np.array([frame[doppler_idx, range_idx] 
                                     for frame in rd_map_complex_history])
            
            # Method 1: Rotor Blade Flash Detection
            blade_flash_score = detect_rotor_blade_flashes(target_history, frame_rate)
            
            # Method 2: Harmonic Analysis for propeller frequencies
            harmonic_score = analyze_propeller_harmonics(target_history, frame_rate)
            
            # Method 3: Spectral Spread Analysis
            spectral_spread_score = analyze_spectral_spread(target_history, frame_rate)
            
            # Method 4: Micro-motion Pattern Recognition
            micro_motion_score = detect_micro_motion_patterns(target_history, frame_rate)
            
            # Combined scoring system
            combined_score = (blade_flash_score * 0.3 + 
                            harmonic_score * 0.25 + 
                            spectral_spread_score * 0.25 + 
                            micro_motion_score * 0.2)
            
            # Classification thresholds
            if combined_score > 0.8:
                drone_type = "DRONE_CONFIRMED"
                confidence = min(0.95, combined_score)
            elif combined_score > 0.6:
                drone_type = "DRONE_PROBABLE"
                confidence = combined_score
            elif combined_score > 0.4:
                drone_type = "DRONE_POSSIBLE"
                confidence = combined_score
            else:
                continue  # Not a drone
            
            # Calculate position
            range_m = range_idx * (range_m_axis[1] - range_m_axis[0])
            velocity_ms = velocity_axis[doppler_idx]
            
            drone_detections.append({
                'type': drone_type,
                'confidence': confidence,
                'position': [0, range_m, 1.5],  # Assume 1.5m height
                'velocity': velocity_ms,
                'blade_flash_score': blade_flash_score,
                'harmonic_score': harmonic_score,
                'spectral_spread': spectral_spread_score,
                'micro_motion': micro_motion_score,
                'signal_strength': signal_strength
            })
            
        except Exception as e:
            logger.warning(f"Drone analysis failed for target {doppler_idx},{range_idx}: {e}")
            continue
    
    return drone_detections

def detect_rotor_blade_flashes(target_history, frame_rate):
    """Detect periodic blade flashes characteristic of rotorcraft"""
    if len(target_history) < 20:
        return 0.0
    
    # Analyze magnitude variations for periodic flashes
    magnitude_history = np.abs(target_history)
    
    # Detrend the signal
    detrended = magnitude_history - np.mean(magnitude_history)
    
    # Look for periodic components in 10-100 Hz range (typical rotor frequencies)
    freqs = np.fft.fftfreq(len(detrended), 1/frame_rate)
    fft_mag = np.abs(np.fft.fft(detrended))
    
    # Focus on rotor frequency range
    rotor_freq_mask = (np.abs(freqs) >= 10) & (np.abs(freqs) <= 100)
    rotor_power = np.sum(fft_mag[rotor_freq_mask])
    total_power = np.sum(fft_mag)
    
    if total_power > 0:
        flash_score = rotor_power / total_power
        return min(1.0, flash_score * 5)  # Scale and cap at 1.0
    
    return 0.0

def analyze_propeller_harmonics(target_history, frame_rate):
    """Analyze harmonic structure typical of propeller-driven aircraft"""
    if len(target_history) < 20:
        return 0.0
    
    # Phase analysis for harmonic content
    phase_history = np.angle(target_history)
    phase_unwrapped = np.unwrap(phase_history)
    
    # Look for harmonic relationships in phase modulation
    freqs = np.fft.fftfreq(len(phase_unwrapped), 1/frame_rate)
    phase_fft = np.abs(np.fft.fft(phase_unwrapped))
    
    # Find fundamental frequency
    fundamental_mask = (np.abs(freqs) >= 5) & (np.abs(freqs) <= 50)
    if not np.any(fundamental_mask):
        return 0.0
    
    fundamental_idx = np.argmax(phase_fft[fundamental_mask])
    fundamental_freq = freqs[fundamental_mask][fundamental_idx]
    
    # Check for harmonics at 2f, 3f, 4f
    harmonic_strength = 0.0
    for harmonic in [2, 3, 4]:
        harmonic_freq = fundamental_freq * harmonic
        harmonic_mask = (np.abs(freqs - harmonic_freq) < 2)  # 2 Hz tolerance
        if np.any(harmonic_mask):
            harmonic_strength += np.max(phase_fft[harmonic_mask])
    
    # Normalize by fundamental strength
    fundamental_strength = np.max(phase_fft[fundamental_mask])
    if fundamental_strength > 0:
        harmonic_score = harmonic_strength / (fundamental_strength * 3)  # 3 harmonics
        return min(1.0, harmonic_score)
    
    return 0.0

def analyze_spectral_spread(target_history, frame_rate):
    """Analyze spectral spreading characteristic of small UAVs"""
    if len(target_history) < 20:
        return 0.0
    
    # Compute spectrogram for time-frequency analysis (complex data)
    f, t, Sxx = stft(target_history, fs=frame_rate, nperseg=min(16, len(target_history)//2), return_onesided=False)
    
    # Analyze spectral spread over time
    spectral_centroids = []
    spectral_spreads = []
    
    for time_slice in range(Sxx.shape[1]):
        power_spectrum = np.abs(Sxx[:, time_slice])
        if np.sum(power_spectrum) > 0:
            # Spectral centroid
            centroid = np.sum(f * power_spectrum) / np.sum(power_spectrum)
            spectral_centroids.append(centroid)
            
            # Spectral spread
            spread = np.sqrt(np.sum(((f - centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
            spectral_spreads.append(spread)
    
    if len(spectral_spreads) > 0:
        avg_spread = np.mean(spectral_spreads)
        spread_variation = np.std(spectral_spreads)
        
        # Drones typically have moderate, varying spectral spread
        if 1.0 < avg_spread < 10.0 and spread_variation > 0.5:
            return min(1.0, (avg_spread * spread_variation) / 50.0)
    
    return 0.0

def detect_micro_motion_patterns(target_history, frame_rate):
    """Detect micro-motion patterns characteristic of hovering/maneuvering drones"""
    if len(target_history) < 20:
        return 0.0
    
    # Analyze short-term velocity variations
    velocity_variations = np.abs(np.diff(target_history))
    
    # Look for characteristic hover patterns (small, rapid variations)
    hover_pattern_score = 0.0
    if len(velocity_variations) > 10:
        # Compute autocorrelation to find repetitive patterns
        autocorr = np.correlate(velocity_variations, velocity_variations, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            
            # Look for periodicity in 0.5-5 second range (typical hover adjustments)
            period_samples = np.arange(int(0.5 * frame_rate), int(5 * frame_rate))
            period_samples = period_samples[period_samples < len(autocorr)]
            
            if len(period_samples) > 0:
                max_autocorr = np.max(autocorr[period_samples])
                hover_pattern_score = max_autocorr
    
    return min(1.0, hover_pattern_score)

def detect_micro_doppler_drone_signatures(rd_map_complex_history, target_indices, frame_rate=10):
    """
    ENHANCED: Real-time micro-Doppler detection for fans, drones, and rotating objects
    """
    if len(rd_map_complex_history) < 10:  # Reduced for better responsiveness
        return []

    drone_detections = []
    
    # ENHANCED: More sensitive target filtering for fan detection
    strong_targets = []
    for doppler_idx, range_idx in zip(target_indices[0], target_indices[1]):
        if (doppler_idx >= len(rd_map_complex_history[0]) or 
            range_idx >= len(rd_map_complex_history[0][0])):
            continue
            
        # Check signal strength - LOWERED threshold for fan detection
        latest_frame = rd_map_complex_history[-1]
        signal_strength = np.abs(latest_frame[doppler_idx, range_idx])
        
        # ENHANCED: More sensitive for detecting fans and rotating objects
        if signal_strength > 5e-4:  # Much lower threshold for fan detection
            strong_targets.append((doppler_idx, range_idx))
    
    # ENHANCED: Real micro-Doppler analysis for each strong target
    for doppler_idx, range_idx in strong_targets:
        try:
            # Extract time series for this target
            target_history = []
            for frame in rd_map_complex_history[-20:]:  # Last 20 frames
                if (doppler_idx < len(frame) and range_idx < len(frame[0])):
                    target_history.append(frame[doppler_idx, range_idx])
            
            if len(target_history) < 10:
                continue
                
            # Perform micro-Doppler analysis
            target_array = np.array(target_history)
            
            # STFT for time-frequency analysis (micro-Doppler signature, complex data)
            f, t, Zxx = stft(target_array, fs=frame_rate, nperseg=8, noverlap=4, return_onesided=False)
            
            # Look for characteristic fan/drone frequencies
            power_spectrum = np.abs(Zxx) ** 2
            freq_profile = np.mean(power_spectrum, axis=1)
            
            # FAN DETECTION: Look for 30-120 Hz signatures (typical fan blade rates)
            fan_freq_range = np.where((f >= 20) & (f <= 120))[0]
            if len(fan_freq_range) > 0:
                fan_power = np.max(freq_profile[fan_freq_range])
                peak_freq_idx = fan_freq_range[np.argmax(freq_profile[fan_freq_range])]
                peak_freq = f[peak_freq_idx]
                
                # Check for sufficient power and characteristic fan signature
                if fan_power > np.mean(freq_profile) * 3:  # 3x above average
                    drone_detections.append({
                        'doppler_idx': doppler_idx,
                        'range_idx': range_idx,
                        'blade_frequency': peak_freq,
                        'confidence': min(0.95, fan_power / (np.mean(freq_profile) * 5)),
                        'type': 'FAN_DETECTED' if peak_freq < 60 else 'DRONE_DETECTED',
                        'micro_doppler_strength': fan_power,
                        'signature_type': 'rotating_blades'
                    })
                    
        except Exception as e:
            logger.debug(f"Micro-Doppler analysis failed for target ({doppler_idx}, {range_idx}): {e}")
            continue
    
    return drone_detections

def detect_micro_doppler_demo_fan(rd_map_complex_history):
    """Demo function for table fan detection with simulated micro-Doppler"""
    current_time = time.time()
    
    # Simulate demo fan detection with time-varying parameters
    if int(current_time) % 10 < 7:  # DETECTION PERIOD (7 out of 10 seconds)
        # Simulate blade frequency with slight variation
        base_freq = 47.0  # Table fan base frequency
        freq_variation = np.sin(current_time * 0.3) * 3.0
        final_blade_freq = base_freq + freq_variation
        
        confidence_base = 0.85
        confidence_variation = np.sin(current_time * 1.5) * 0.1
        final_confidence = confidence_base + confidence_variation
        
        return [{
            'range': 1.8 + np.sin(current_time * 0.5) * 0.2,  # Slight range variation
            'velocity': np.sin(current_time * 0.8) * 0.1,  # Small velocity variation
            'type': 'DRONE_CONFIRMED',
            'drone_class': "Table Fan",
            'blade_frequency': final_blade_freq,
            'confidence': final_confidence,
            'threat_level': "LOW",
            'propeller_count': 3
        }]
    else:
        # NO DETECTION PERIOD
        return []

def classify_micro_doppler_target(micro_doppler_signature):
    """
    Distinguish between birds and drones using micro-Doppler characteristics.
    Based on research: birds have 5-15 Hz wing flaps, drones have 20-200 Hz blade flash.
    """
    freq = micro_doppler_signature.get('blade_frequency', 0)
    power_ratio = micro_doppler_signature.get('power_ratio', 0)
    
    # Classification logic based on research
    if freq < 15:
        return {
            'classification': 'BIRD',
            'confidence': 0.8,
            'reasoning': 'Low frequency wing flap pattern'
        }
    elif freq >= 20 and power_ratio > 0.3:
        return {
            'classification': 'DRONE',
            'confidence': 0.9,
            'reasoning': 'High frequency propeller blade flash'
        }
    else:
        return {
            'classification': 'UNKNOWN',
            'confidence': 0.3,
            'reasoning': 'Ambiguous micro-Doppler signature'
        }

def coherent_micro_doppler_integration(frame_history, integration_time=2.0):
    """
    Integrate multiple frames coherently to enhance micro-Doppler signatures.
    Critical for detecting weak drone signatures.
    """
    if len(frame_history) < int(integration_time * 10):  # 10 Hz frame rate
        return None
    
    # Coherently sum complex data
    integrated_frame = np.zeros_like(frame_history[0], dtype=complex)
    for frame in frame_history[-int(integration_time * 10):]:
        integrated_frame += frame
    
    return integrated_frame / len(frame_history)

def adaptive_small_drone_detection(rd_map_db, micro_doppler_signatures):
    """
    Lower detection thresholds for confirmed micro-Doppler drone signatures.
    """
    enhanced_threshold_map = np.full_like(rd_map_db, -70.0)  # Standard threshold
    
    for drone_sig in micro_doppler_signatures:
        if drone_sig['confidence'] > 0.7:
            # Lower threshold around confirmed drone signatures
            range_idx = np.argmin(np.abs(range_m_axis - drone_sig['range']))
            doppler_idx = np.argmin(np.abs(velocity_axis - drone_sig['velocity']))
            
            # Create detection window around drone signature
            r_window = slice(max(0, range_idx-2), min(len(range_m_axis), range_idx+3))
            d_window = slice(max(0, doppler_idx-2), min(len(velocity_axis), doppler_idx+3))
            
            # Lower threshold by 15 dB for confirmed drones
            enhanced_threshold_map[d_window, r_window] -= 15
    
    return enhanced_threshold_map

def apply_temporal_coherence_filter(frame_history):
    """
    Apply temporal coherence to improve detection reliability.
    Filters out points that don't appear consistently across multiple frames.
    """
    if len(frame_history) < 3:
        return frame_history[-1] if frame_history else pd.DataFrame()
    
    # Combine all frames
    all_frames = pd.concat(frame_history, ignore_index=True)
    
    if all_frames.empty:
        return pd.DataFrame()
    
    # Create spatial grid for coherence analysis
    grid_size = 0.5  # 50cm grid
    all_frames['grid_x'] = (all_frames['x'] / grid_size).round().astype(int)
    all_frames['grid_y'] = (all_frames['y'] / grid_size).round().astype(int)
    all_frames['grid_z'] = (all_frames['z'] / grid_size).round().astype(int)
    
    # Count occurrences in each grid cell
    grid_counts = all_frames.groupby(['grid_x', 'grid_y', 'grid_z']).size()
    
    # Keep points that appear in at least 40% of frames
    min_occurrences = max(1, len(frame_history) * 0.4)
    persistent_grids = grid_counts[grid_counts >= min_occurrences].index
    
    # Filter current frame to only include persistent points
    current_frame = frame_history[-1].copy()
    
    if current_frame.empty:
        return pd.DataFrame()
        
    current_frame['grid_x'] = (current_frame['x'] / grid_size).round().astype(int)
    current_frame['grid_y'] = (current_frame['y'] / grid_size).round().astype(int)
    current_frame['grid_z'] = (current_frame['z'] / grid_size).round().astype(int)
    
    # Keep only temporally coherent points
    coherent_mask = current_frame.apply(
        lambda row: (row['grid_x'], row['grid_y'], row['grid_z']) in persistent_grids, 
        axis=1
    )
    
    # Drop grid columns only if they exist, ignore errors if they don't
    return current_frame[coherent_mask].drop(['grid_x', 'grid_y', 'grid_z'], axis=1, errors='ignore')

def coherent_multipath_suppression(range_doppler_history: list, coherence_threshold: float = 0.8):
    """
    Suppresses multipath ghosts by rewarding temporally coherent signals.
    This should be applied before peak detection to eliminate false targets.

    Args:
        range_doppler_history (list): A deque of recent complex range-Doppler maps.
        coherence_threshold (float): A value between 0 and 1. Higher values are stricter.

    Returns:
        np.ndarray: The latest range-Doppler map with multipath signals suppressed.
    """
    if len(range_doppler_history) < 3:
        return range_doppler_history[-1] if range_doppler_history else np.array([])

    # Stack the last 3 frames
    frames = np.stack(list(range_doppler_history)[-3:], axis=0)
    
    # Calculate the phase consistency over time for each pixel
    # A stable, direct-path target will have a more consistent phase progression
    phase_diffs = np.angle(frames[1:] * np.conj(frames[:-1]))
    phase_std_dev = np.std(phase_diffs, axis=0)

    # Create a coherence mask. Low standard deviation means high coherence.
    # We normalize by a small value to handle near-zero std dev
    coherence_map = 1 - (phase_std_dev / (np.pi/2))
    coherence_map = np.clip(coherence_map, 0, 1)

    # Get the latest frame
    latest_frame = frames[-1].copy()

    # Attenuate signals in areas of low coherence
    suppression_factor = 0.35  # Penalize incoherent signals but not too harshly
    latest_frame[coherence_map < coherence_threshold] *= suppression_factor
    
    return latest_frame

# === TEMPORARY ETHERNET DEBUGGING ===
ETHERNET_DEBUG_MODE = False  # Set to False to return to WiFi
ETHERNET_PI_IP = "192.168.100.2"  # Pi's ethernet IP
ETHERNET_LAPTOP_IP = "192.168.100.1"  # Laptop's ethernet IP

def get_active_network_config():
    """Automatically detect which network to use."""
    if ETHERNET_DEBUG_MODE:
        logger.info("ETHERNET DEBUG MODE: Using wired connection")
        return {
            'pi_ip': ETHERNET_PI_IP,
            'laptop_bind_ip': ETHERNET_LAPTOP_IP,
            'connection_type': 'ETHERNET'
        }
    else:
        logger.info("HOTSPOT MODE: Using hotspot connection")
        return {
            'pi_ip': "192.168.137.62",  # Pi's hotspot IP
            'laptop_bind_ip': "0.0.0.0",  # Bind to all interfaces
            'connection_type': 'HOTSPOT'
        }

ENABLE_TEST_DATA = False  # Set to False by default - controlled by GUI toggle
CURRENT_DATA_SOURCE = 'real'  # Global variable to track current data source setting

def generate_test_data(enable_synthetic=None):
    """Generate REALISTIC test radar data with proper micro-Doppler signatures"""
    global CURRENT_DATA_SOURCE
    if enable_synthetic is None:
        enable_synthetic = (CURRENT_DATA_SOURCE == 'synthetic')
        
    if not enable_synthetic:
        return []  # Skip test data generation when disabled
    current_time = time.time()
    test_points = []
    
    # REALISTIC indoor targets with proper noise characteristics
    targets = [
        # Wall - strong, static, many points
        {'type': 'wall', 'x': -1.8, 'y': 3.5, 'z': 0.0, 'velocity': 0.0, 'snr': -45, 'points': 35, 'micro_doppler': None},
        
        # Table - medium strength, static  
        {'type': 'table', 'x': 0.8, 'y': 2.2, 'z': -0.4, 'velocity': 0.0, 'snr': -62, 'points': 15, 'micro_doppler': None},
        
        # Person - medium strength, moving
        {'type': 'person', 'x': -0.5, 'y': 1.8, 'z': 0.3, 'velocity': 0.4, 'snr': -68, 'points': 12, 'micro_doppler': None},
        
        # SIMULATED FAN - with micro-Doppler signature!
        {'type': 'fan', 'x': 1.2, 'y': 1.5, 'z': 0.8, 'velocity': 0.0, 'snr': -58, 'points': 8, 
         'micro_doppler': {'type': 'DRONE_CONFIRMED', 'blade_frequency': 45.0, 'confidence': 0.85}},
    ]
    
    for target in targets:
        # Add realistic noise and scatter
        for i in range(target['points']):
            # More realistic noise distribution
            noise_scale = 0.15 if target['type'] == 'wall' else 0.08
            noise_x = target['x'] + np.random.normal(0, noise_scale)
            noise_y = target['y'] + np.random.normal(0, noise_scale) 
            noise_z = target['z'] + np.random.normal(0, noise_scale/2)
            
            # Realistic velocity variation
            velocity_noise = np.random.normal(0, 0.05)
            final_velocity = target['velocity'] + velocity_noise
            
            # Realistic SNR variation
            snr_noise = np.random.normal(0, 4)
            final_snr = target['snr'] + snr_noise
            
            # Calculate realistic parameters
            range_val = np.sqrt(noise_x**2 + noise_y**2 + noise_z**2)
            azimuth = np.degrees(np.arctan2(noise_x, noise_y))
            
            point = {
                'x': noise_x, 'y': noise_y, 'z': noise_z,
                'velocity': final_velocity,
                'snr': final_snr,
                'range': range_val,
                'azimuth': azimuth,
                'timestamp': current_time,
                'scan_count': 1
            }
            
            # ADD MICRO-DOPPLER PROPERTIES FOR FAN
            if target['micro_doppler']:
                point['micro_doppler_type'] = target['micro_doppler']['type']
                point['drone_class'] = 'Table Fan'
                point['blade_frequency'] = target['micro_doppler']['blade_frequency'] + np.random.normal(0, 2)
                point['threat_level'] = 'LOW'
                point['drone_confidence'] = target['micro_doppler']['confidence']
                point['propeller_count'] = 3
            
            test_points.append(point)
    
    return test_points

# Get current network config
NETWORK_CONFIG = get_active_network_config()



# ==== SECTION: Dash App Layout & Callbacks ====
server = Flask(__name__)
app = Dash(__name__, server=server, suppress_callback_exceptions=True)
app.title = "Fixed 4D Radar Dashboard - CN0566"

# ==============================================================================
# ==== SECTION: Master HTML Template & CSS =====================================
# ==============================================================================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg-main: #121212;
                --bg-sidebar: #1E1E1E;
                --bg-card: #282828;
                --border-color: #383838;
                --text-primary: #E0E0E0;
                --text-secondary: #B3B3B3;
                --accent-color: #1DB954;
            }
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                margin: 0; padding: 0; background-color: var(--bg-main); color: var(--text-primary);
                overflow: hidden;
            }
            h1, h2 { margin: 0 0 15px 0; font-weight: 600; }
            h1 { font-size: 1.4rem; color: white; }
            h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); }
            label { font-size: 0.9rem; font-weight: 500; display: block; margin: 15px 0 5px 0; }

            .app-container {
                display: flex;
                height: 100vh;
                width: 100vw;
            }
            .main-view {
                flex-grow: 1;
                height: 100vh;
                position: relative;
                transition: width 0.3s ease-in-out;
                width: calc(100% - 480px);
                z-index: 1;
            }
            .point-cloud-graph { height: 100% !important; width: 100% !important; }

            .sidebar {
                width: 480px;
                min-width: 480px;
                height: 100vh;
                background-color: var(--bg-sidebar);
                padding: 20px;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
                border-left: 1px solid var(--border-color);
                transition: all 0.3s ease-in-out;
                z-index: 10;
            }

            .sidebar-header { border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 15px; }
            .status-bar { display: flex; align-items: center; margin-top: 10px; font-size: 0.9rem; }
            .status-indicator { width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; transition: background-color 0.5s ease; }
            .status-ok { background-color: #28a745; box-shadow: 0 0 8px #28a745; }
            .status-warning { background-color: #ffc107; box-shadow: 0 0 8px #ffc107; }
            .status-error { background-color: #dc3545; box-shadow: 0 0 8px #dc3545; }

            .sidebar-card { background-color: var(--bg-card); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
            .progress-container { width: 100%; height: 10px; background-color: #444; border-radius: 5px; overflow: hidden; position: relative; }
            .progress-fill { height: 100%; background: var(--accent-color); transition: width 0.3s ease-in-out; }
            .progress-text { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); font-size: 0.7rem; font-weight: bold; color: white; text-shadow: 1px 1px 2px black; }

            .metric-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin-top: 10px;
            }

            .metric-item {
                background-color: #333;
                padding: 8px;
                border-radius: 6px;
                text-align: center;
                min-height: 50px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .metric-value { font-size: 1.1em; font-weight: bold; color: var(--accent-color); }
            .metric-label { font-size: 0.75em; opacity: 0.8; }

            .selection-details {
                background-color: #2a2a2a;
                border-radius: 6px;
                padding: 12px;
                margin-top: 10px;
                border-left: 3px solid var(--accent-color);
            }

            .coordinate-display {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 8px;
                margin-top: 8px;
            }

            .coordinate-item {
                background-color: #1a1a1a;
                padding: 6px;
                border-radius: 4px;
                text-align: center;
                font-family: 'Courier New', monospace;
            }

            .tabs-content {
                padding: 5px 0 0 0;
                background-color: transparent;
                display: flex;
                flex-direction: column;
                min-height: 300px;
            }

            .tabs-content .graph {
                flex-grow: 1;
                height: 280px;
            }

            .rc-slider-track { background-color: var(--accent-color) !important; }
            .rc-slider-handle { border-color: var(--accent-color) !important; box-shadow: 0 0 5px var(--accent-color); }

            .Select-control {
                background-color: #333 !important;
                border-color: var(--border-color) !important;
                color: var(--text-primary) !important;
                box-shadow: none !important;
            }

            .Select--single > .Select-control .Select-value {
                background-color: transparent !important;
                color: var(--text-primary) !important;
                line-height: 32px !important;
            }

            .Select-value-label {
                color: var(--text-primary) !important;
            }

            .Select-placeholder {
                color: var(--text-secondary) !important;
            }

            .Select-arrow-zone {
                color: var(--text-primary) !important;
            }

            .Select-arrow {
                border-color: var(--text-primary) transparent transparent !important;
            }

            .Select-menu-outer {
                background-color: #333 !important;
                border-color: var(--border-color) !important;
                border-top: none !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            }

            .Select-menu {
                background-color: #333 !important;
            }

            .Select-option {
                background-color: #333 !important;
                color: var(--text-primary) !important;
                padding: 8px 12px !important;
            }

            .Select-option.is-focused {
                background-color: #444 !important;
                color: var(--text-primary) !important;
            }

            .Select-option.is-selected {
                background-color: var(--accent-color) !important;
                color: white !important;
            }

            .Select-option.is-selected.is-focused {
                background-color: var(--accent-color) !important;
                color: white !important;
            }

            .Select-input {
                color: var(--text-primary) !important;
            }

            .Select-input > input {
                color: var(--text-primary) !important;
            }

            .diagnostic-tabs { border-bottom: 1px solid var(--border-color); }
            .diagnostic-tab { background-color: transparent !important; border: none !important; color: var(--text-secondary) !important; padding: 12px !important; transition: background-color 0.2s, color 0.2s; border-bottom: 3px solid transparent !important; }
            .diagnostic-tab--selected { background-color: transparent !important; color: var(--accent-color) !important; font-weight: bold; border-bottom: 3px solid var(--accent-color) !important; }

            .sidebar-toggle {
                position: absolute; top: 15px; left: 15px; z-index: 1000; background-color: rgba(40, 40, 40, 0.7);
                color: white; border: 1px solid #444; border-radius: 50%; width: 40px; height: 40px;
                cursor: pointer; font-size: 1.5rem; line-height: 1; transition: all 0.2s;
            }
            .status-heartbeat {
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7);
                }
                70% {
                    transform: scale(1);
                    box-shadow: 0 0 0 10px rgba(40, 167, 69, 0);
                }
                100% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(40, 167, 69, 0);
                }
            }
            .sidebar-toggle:hover { background-color: rgba(60, 60, 60, 0.9); }
            .reset-camera-btn {
                position: absolute; top: 15px; right: 15px; z-index: 1000; 
                background-color: rgba(40, 40, 40, 0.7); color: white; 
                border: 1px solid #444; border-radius: 50%; width: 40px; height: 40px;
                cursor: pointer; font-size: 1.2rem; line-height: 1; transition: all 0.2s;
                display: flex; align-items: center; justify-content: center;
            }
            .reset-camera-btn:hover { 
                background-color: rgba(60, 60, 60, 0.9); 
                transform: scale(1.1);
                box-shadow: 0 0 10px rgba(29, 185, 84, 0.5);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Enhanced sidebar layout with click information
app.layout = html.Div(className="app-container", children=[
    html.Div(id="main-view", className="main-view", children=[
        dcc.Graph(
            id="point-cloud-3d",
            className="point-cloud-graph",
            config={'displayModeBar': False}
        ),
        html.Button("<", id="sidebar-toggle-btn", className="sidebar-toggle"),
        html.Button("🎯", id="reset-camera-btn", className="reset-camera-btn", title="Reset Camera View")
    ]),
    html.Div(id="sidebar", className="sidebar", children=[
        html.Div(className="sidebar-header", children=[
            html.H1("4D Radar Dashboard", className="sidebar-title"),
            html.Div(className="status-bar", children=[
                html.Div(id="status-indicator", className="status-indicator"),
                html.Span(id="status-text", children="Initializing...")
            ])
        ]),

        html.Div(className="sidebar-card", children=[
            html.H2("Scan Progress"),
            html.Div(className="progress-container", children=[
                html.Div(id="progress-fill", className="progress-fill"),
                html.Span(id="progress-text", children="0%", className="progress-text")
            ])
        ]),

        html.Div(className="sidebar-card", children=[
            html.H2("Display & Environment"),
            
            html.Label("Data Source:"),
            dcc.RadioItems(
                id='data-source-radio',
                options=[
                    {'label': 'Real Data from Pi', 'value': 'real'},
                    {'label': 'Test Data (Synthetic)', 'value': 'synthetic'},
                ],
                value='real',  # Default to real data
                labelStyle={'display': 'block', 'margin-bottom': '5px'},
                style={'margin-bottom': '10px'},
            ),
            
            # Connection status indicator
            html.Div(id='connection-status', children=[
                html.Span("⚪ Checking connection...", style={'color': 'yellow', 'font-size': '14px'})
            ], style={'margin-bottom': '20px'}),

            html.Label("Environment Mode:"),
            dcc.RadioItems(
                id='environment-mode-radio',
                options=[
                    {'label': 'Indoor (Strict Filter)', 'value': 'indoor'},
                    {'label': 'Outdoor (Standard Filter)', 'value': 'outdoor'},
                ],
                value='indoor',
                labelStyle={'display': 'block', 'margin-bottom': '5px'},
                style={'margin-bottom': '20px'}
            ),

            html.Label("Point Cloud Color Mode:"),
            dcc.Dropdown(
                id="display-mode-dropdown",
                options=[
                    {'label': 'Color by Velocity', 'value': 'velocity'},
                    {'label': 'Color by Range', 'value': 'range'},
                    {'label': 'Color by Signal Strength', 'value': 'signal_strength'}
                ],
                value='velocity',
                clearable=False,
                style={'margin-bottom': '15px'}
            ),

            html.Label("Visualization Options:"),
            dcc.Checklist(
                id='filter-options-checklist',
                options=[
                    {'label': 'Doppler Mode (Micro-Doppler)', 'value': 'DOPPLER'},
                    {'label': 'Show Semantic Voxels', 'value': 'SURFACE'},
                    {'label': 'Show Raw Points (Debug)', 'value': 'POINTS'},
                ],
                value=['SURFACE', 'DOPPLER'],  # Default with semantic voxels
                style={'margin-bottom': '20px'}
            ),

            html.Label("Performance Mode:"),
            dcc.Dropdown(
                id="performance-mode-dropdown",
                options=[
                    {'label': 'Balanced (Recommended)', 'value': 'balanced'},
                    {'label': 'Full Quality', 'value': 'full'},
                    {'label': 'Lightweight', 'value': 'minimal'},
                ],
                value='balanced',  # Default to balanced instead of auto
                clearable=False,
                style={'margin-bottom': '20px'}
            ),

            html.Label("Point Size:"),
            dcc.Slider(
                id="point-size-slider",
                min=1, max=8, step=1, value=3,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),

            html.Label("Voxel Resolution (m):"),
            dcc.Slider(
                id="voxel-resolution-slider",
                min=0.1, max=1.0, step=0.05, value=0.4,
                marks={i/10: f"{i/10:.1f}" for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),

            html.Label("Voxel Size:"),
            dcc.Slider(
                id="voxel-size-slider",
                min=2, max=15, step=1, value=6, # Default size
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("SNR Threshold (dB):"),
            dcc.Slider(id="snr-threshold-slider", min=-120, max=-40, step=1, value=-60,
                    marks={i: f"{i}" for i in range(-120, -39, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}),
        ]),

        # In your layout, update the diagnostic tabs:
        dcc.Tabs(id="diagnostic-tabs", value='tab-doppler', className="diagnostic-tabs", children=[
            dcc.Tab(label='Range-Doppler', value='tab-doppler', className="diagnostic-tab", selected_className="diagnostic-tab--selected"),
            # --- ADD THESE TWO NEW TABS ---
            dcc.Tab(label='Elevation View', value='tab-elevation', className="diagnostic-tab", selected_className="diagnostic-tab--selected"),
            dcc.Tab(label='Top-Down View', value='tab-top-down', className="diagnostic-tab", selected_className="diagnostic-tab--selected"),
            # --- END OF ADDITION ---
            dcc.Tab(label='Micro-Doppler', value='tab-micro-doppler', className="diagnostic-tab", selected_className="diagnostic-tab--selected"),
            dcc.Tab(label='Range History', value='tab-waterfall', className="diagnostic-tab", selected_className="diagnostic-tab--selected"),
        ]),

        html.Div(id='tabs-content', className="tabs-content"),

        html.Div(className="sidebar-card", children=[
            html.H2("Micro-Doppler Analysis"),
            html.Div(id="micro-doppler-status"),
            html.Div(className="metric-grid", children=[
                html.Div(id="blade-frequency-display", className="metric-item"),
                html.Div(id="drone-confidence-display", className="metric-item"),
                html.Div(id="signature-strength-display", className="metric-item"),
                html.Div(id="detection-type-display", className="metric-item"),
            ])
        ]),        

        html.Div(className="sidebar-card", children=[
            html.H2("Detected Objects"),
            dash_table.DataTable(
                id='object-table',
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Status", "id": "status"}, # ADDED
                    {"name": "Points", "id": "points"},
                    {"name": "Range (m)", "id": "range"},
                    {"name": "Velocity (m/s)", "id": "velocity"},
                ],
                style_cell={'backgroundColor': '#282828', 'color': 'white', 'textAlign': 'left'},
                style_header={'backgroundColor': '#1DB954', 'fontWeight': 'bold'},
            )
        ]),        

        html.Div(id="selection-info-card", className="sidebar-card", children=[
            html.H2("Point Information"),
            html.P("Click on a point to see details", style={'color': '#B3B3B3', 'font-style': 'italic'})
        ]),

        html.Div(className="sidebar-card", children=[
             html.H2("Environment Status"),
             html.Div(className="metric-grid", children=[
                html.Div(id="point-count-display", className="metric-item"),
                html.Div(id="current-az-display", className="metric-item"),
                html.Div(id="objects-detected-display", className="metric-item"),
                html.Div(id="scan-coverage-display", className="metric-item"),
             ])
        ]),

        html.Div(className="sidebar-card", children=[
            html.H2("System Performance"),
            html.Div(id="system-metrics")
        ]),
    ]),

    dcc.Interval(id="fast-update-interval", interval=CONFIG['dashboard']['FAST_INTERVAL'], n_intervals=0),
    dcc.Interval(id="slow-update-interval", interval=CONFIG['dashboard']['SLOW_INTERVAL'], n_intervals=0),
    dcc.Store(id='sidebar-state-store', data={'expanded': True}),
    dcc.Store(id='processed-radar-data'),
    dcc.Store(id='kalman-tracker-data'),
    dcc.Store(id='micro-doppler-data'),

])

ENABLE_OBJECT_DETECTION = True

# ==============================================================================
# ==== SECTION: Callbacks ====================================================
# ==============================================================================
@app.callback(
    [Output("point-cloud-3d", "figure"),
     Output("point-count-display", "children"),
     Output("current-az-display", "children"),
     Output("objects-detected-display", "children"),
     Output("scan-coverage-display", "children"),
     Output("object-table", "data"),
     Output("system-metrics", "children")],
    [Input("fast-update-interval", "n_intervals"),
     Input("display-mode-dropdown", "value"),
     Input("point-size-slider", "value"),
     Input("snr-threshold-slider", "value"),
     Input("environment-mode-radio", "value"),
     Input("filter-options-checklist", "value"),
     Input("performance-mode-dropdown", "value"),
     Input("voxel-resolution-slider", "value"),
     Input("voxel-size-slider", "value"),
     Input("data-source-radio", "value")],  # ADDED: This Input was missing
    [State("point-cloud-3d", "relayoutData")]
)
def update_point_cloud(n, display_mode, point_size, snr_threshold, environment_mode, filter_options, performance_mode, voxel_resolution, voxel_size, data_source, relayout_data): # ADDED: The 'data_source' argument was missing
    """
    COMPLETE ENHANCED VERSION WITH MICRO-DOPPLER DRONE DETECTION + TEMPORAL COHERENCE (June 22, 2025):
    - PRESERVED: All original functionality including Kalman tracking, bounding boxes, velocity vectors
    - FIXED: Oversensitivity issues and false positives
    - ENHANCED: Drone highlighting with micro-Doppler signatures
    - ENHANCED: Temporal coherence for improved detection
    - ENHANCED: Doppler-specific filtering modes
    """
    global NEXT_OBJECT_ID, KALMAN_TRACKERS
    start_time = time.time()
    
    # Initialize Kalman trackers dictionary if not exists
    if 'KALMAN_TRACKERS' not in globals():
        global KALMAN_TRACKERS
        KALMAN_TRACKERS = {}
    
    # Determine actual performance mode
    if performance_mode == 'auto':
        actual_performance_mode = get_performance_mode()
    else:
        actual_performance_mode = performance_mode
    
    # Optimize Kalman trackers based on performance mode
    optimize_kalman_trackers(actual_performance_mode)
    
    filter_options = filter_options if filter_options else []
    CONFIDENCE_THRESHOLD = 3
    PRELIMINARY_COLOR = 'rgb(128, 128, 128)'
    CONFIRMED_COLORS = {"Wall/Large Surface": "cyan", "Furniture/Cabinet": "orange", "Small Object": "yellow", 
                       "Static Structure": "lightblue", "Drone/UAV": "magenta", "Person": "red", 
                       "Moving Object": "lime", "Unknown Object": "gray"}
    
    # Performance-based limits - MICRO-DOPPLER ALWAYS ENABLED
    if actual_performance_mode == 'minimal':
        max_display_points = 800
        surface_opacity = 0.5
        skip_dense_cloud = True
        kalman_frequency = 5
        micro_doppler_enabled = True  # Still enabled in minimal mode
    elif actual_performance_mode == 'balanced':
        max_display_points = 1500
        surface_opacity = 0.7
        skip_dense_cloud = False
        kalman_frequency = 2  # Increased frequency for better micro-Doppler
        micro_doppler_enabled = True
    else:  # 'full'
        max_display_points = 3000
        surface_opacity = 0.7
        skip_dense_cloud = False
        kalman_frequency = 1
        micro_doppler_enabled = True
    
    # Fixed opacity values
    grid_opacity = 0.1
    cone_opacity = 0.15
    
    # === GET RADAR DATA ===
    if data_source == 'synthetic':
        # User has selected synthetic data mode
        all_points_list = generate_test_data(enable_synthetic=True)
        # Mock other values so the dashboard doesn't crash
        current_az, scan_count = 0, 1
        last_heartbeat = time.time()
        drone_detections, last_drone_detection = 0, 0
        # In synthetic mode, profile defaults to indoor for stable parameters
        profile = INDOOR_PROFILE 
    else:
        # User has selected real data mode
        with RADAR_DATA_LOCK:
            # Safely get all real data
            all_points_list = [p for d in RADAR_DATA.get('persistent_points', {}).values() for p in d]
            current_az = RADAR_DATA.get('current_az', 0.0)
            scan_count = RADAR_DATA.get('scan_count', 0)
            last_heartbeat = RADAR_DATA.get('last_heartbeat_time', 0)
            drone_detections = RADAR_DATA.get('drone_detections', 0)
            last_drone_detection = RADAR_DATA.get('last_drone_detection', 0)
            # Get the correct profile for real data
            active_profile_name = RADAR_DATA.get('active_profile_name', 'indoor')
            profile = INDOOR_PROFILE if active_profile_name == 'indoor' else OUTDOOR_PROFILE

    # This check now runs AFTER attempting to load from the selected source
    if not all_points_list:
        logger.warning(f"[WARNING] NO POINTS FOUND for data source: {data_source}")

        max_range = CONFIG['signal_processing']['MAX_RANGE_DISPLAY']
        fig = go.Figure(data=create_radar_cone_traces_with_opacity(max_range, CONFIG['scanning']['AZ_FOV'], cone_opacity))

        # Add a helpful message as a 3D text annotation
        fig.add_trace(go.Scatter3d(
            x=[0], y=[5], z=[5],
            mode='text',
            text=["<b>No Radar Data Available</b><br><br>" +
                "• Check Pi transmitter connection<br>" +
                "• Ensure transmitter script is running<br>" + 
                "• Verify network connectivity"],
            textfont=dict(size=16, color="yellow"),
            showlegend=False,
            hoverinfo='none'
        ))

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#000000", margin=dict(l=0, r=0, b=0, t=0),
            uirevision=f'pointcloud-{str(filter_options)}', # MODIFIED LINE
            showlegend=False,
            scene=dict(
                xaxis=dict(range=[-max_range, max_range]),
                yaxis=dict(range=[0, max_range]),
                zaxis=dict(range=[-max_range/4, max_range/4])
            )
        )

        empty_metrics = [html.Div("0", className="metric-value"), html.Div("Waiting for Data", className="metric-label")]
        return [fig] + [empty_metrics] * 6
    
    # === PROCESS POINTS ===
    df_all = pd.DataFrame(all_points_list)
    
    # Memory-efficient processing
    df_filtered = memory_efficient_point_sampling(df_all, max_points=max_display_points, memory_budget_mb=CONFIG['performance']['POINT_CLOUD_MEMORY_LIMIT_MB'])
    df_filtered = apply_spatial_filtering(remove_statistical_outliers(df_filtered))
    
    if environment_mode == 'indoor': 
        df_filtered = df_filtered[df_filtered['range'] <= 7.5]
    
    # ENHANCED FILTERING WITH DOPPLER MODES - FIXED TO BE MORE LENIENT
    if 'MTI' in filter_options:
        # Traditional MTI (Moving Target Indicator) - More lenient
        df_filtered = df_filtered[np.abs(df_filtered['velocity']) > 0.15]  # Reduced from 0.3
        df_filtered = df_filtered[df_filtered['snr'] > -85]  # More lenient SNR
    elif 'DOPPLER' in filter_options:
        # FIXED: Much more lenient Doppler-specific mode for micro-Doppler detection
        df_filtered = df_filtered[
            (np.abs(df_filtered['velocity']) > 0.05) |  # Very low threshold for movement
            (df_filtered.get('micro_doppler_type', '') == 'DRONE_CONFIRMED') |  # Confirmed drones
            (df_filtered['snr'] > -95)  # Include strong static targets that might have micro-Doppler
        ]
        logger.info(f"[DEBUG] After DOPPLER filtering: {len(df_filtered)}")

    # Apply SNR threshold - More lenient
    df_plot = df_filtered[df_filtered['snr'] > max(snr_threshold, -100)].copy()  # Don't be too strict
    
    current_time = time.time()
    df_recent = df_plot[current_time - df_plot['timestamp'] < 2.0].copy()  # Increased from 1.5s to 2s
    
    # === PRESERVED: TEMPORAL COHERENCE ENHANCEMENT ===
    # Apply temporal coherence to improve drone detection reliability
    if not hasattr(update_point_cloud, 'temporal_history'):
        update_point_cloud.temporal_history = deque(maxlen=10)  # 1 second at 10 Hz
    
    # Store current frame for temporal analysis
    update_point_cloud.temporal_history.append(df_recent.copy())
    
    # Apply temporal coherence filtering
    if len(update_point_cloud.temporal_history) >= 5:
        df_recent = apply_temporal_coherence_filter(list(update_point_cloud.temporal_history))
    
    # === PRESERVED: OPTIMIZED KALMAN TRACKING ===
    new_clusters = []
    if not df_recent.empty and len(df_recent) > profile["MIN_POINTS_FOR_DETECTION"]:
        if n % kalman_frequency == 0:
            new_clusters = detect_objects_dbscan(df_recent, eps=profile["DBSCAN_EPS"], min_samples=profile["MIN_SAMPLES_IN_CLUSTER"])

    # Predict all existing trackers
    for tracker_id, tracker in KALMAN_TRACKERS.items():
        tracker.predict()

    # Associate clusters with existing trackers
    matched_tracker_ids = set()
    for cluster_df in new_clusters[:5]:
        if len(cluster_df) < 3: continue
        
        new_centroid = calculate_centroid(cluster_df)
        bbox_params = calculate_oriented_bounding_box(cluster_df)
        if not bbox_params: continue
        
        best_match_id, min_dist = None, float('inf')
        for tracker_id, tracker in KALMAN_TRACKERS.items():
            if tracker_id in matched_tracker_ids: continue
            
            predicted_pos = tracker.position
            dist = np.linalg.norm(new_centroid - predicted_pos)
            velocity_magnitude = np.linalg.norm(tracker.velocity)
            association_threshold = 1.5 + velocity_magnitude * 0.1
            
            if dist < association_threshold and dist < min_dist:
                min_dist, best_match_id = dist, tracker_id
        
        if best_match_id is not None:
            tracker = KALMAN_TRACKERS[best_match_id]
            
            # Enhanced update with micro-Doppler data
            micro_doppler_data = None
            if 'micro_doppler_type' in cluster_df.columns:
                # Extract micro-Doppler information for this cluster
                micro_doppler_rows = cluster_df[cluster_df['micro_doppler_type'].notna()]
                if not micro_doppler_rows.empty:
                    micro_doppler_data = {
                        'type': micro_doppler_rows['micro_doppler_type'].iloc[0],
                        'confidence': 0.8,  # Default confidence
                        'timestamp': time.time()
                    }
            
            # Use enhanced update method
            if hasattr(tracker, 'update') and len(tracker.update.__code__.co_varnames) > 2:
                # Enhanced tracker
                tracker.update(new_centroid, micro_doppler_data)
            else:
                # Legacy tracker
                tracker.update(new_centroid)
            
            matched_tracker_ids.add(best_match_id)
            
            # Enhanced classification
            guess = classify_cluster(cluster_df, bbox_params)
            if hasattr(tracker, 'classification_history'):
                tracker.classification_history.append(guess)
            
            # Enhanced confirmation logic
            if not tracker.is_confirmed and hasattr(tracker, 'classification_history'):
                if len(tracker.classification_history) >= CONFIDENCE_THRESHOLD:
                    last_n = list(tracker.classification_history)[-CONFIDENCE_THRESHOLD:]
                    if all(g == last_n[0] for g in last_n) and last_n[0] != "Noise":
                        tracker.is_confirmed = True
                        tracker.label = last_n[0]
                        tracker.color = CONFIRMED_COLORS.get(last_n[0], "lime")
        else:
            # Create new tracker with enhanced classification
            guess = classify_cluster(cluster_df, bbox_params)
            if guess != "Noise":
                # Determine target type from cluster data
                target_type = 'UNKNOWN'
                if 'micro_doppler_type' in cluster_df.columns:
                    micro_doppler_types = cluster_df['micro_doppler_type'].dropna()
                    if not micro_doppler_types.empty:
                        target_type = micro_doppler_types.iloc[0]
                elif guess == "Moving":
                    target_type = 'MOVING_OBJECT'
                elif guess == "Static":
                    target_type = 'STATIC_OBJECT'
                
                # Create enhanced tracker
                new_tracker = EnhancedKalmanTracker(new_centroid, NEXT_OBJECT_ID, target_type)
                if hasattr(new_tracker, 'classification_history'):
                    new_tracker.classification_history.append(guess)
                
                KALMAN_TRACKERS[NEXT_OBJECT_ID] = new_tracker
                NEXT_OBJECT_ID += 1
                
                logger.info(f"New enhanced tracker {NEXT_OBJECT_ID-1} created: {target_type} at {new_centroid}")
    
    # Enhanced stale tracker removal with better criteria
    stale_tracker_ids = []
    for tracker_id, tracker in KALMAN_TRACKERS.items():
        # Use enhanced tracker deletion criteria if available
        if hasattr(tracker, 'should_delete'):
            if tracker.should_delete():
                stale_tracker_ids.append(tracker_id)
        else:
            # Legacy deletion logic
            is_static = tracker.is_confirmed and hasattr(tracker, 'label') and "Static" in tracker.label
            max_age = profile.get("STATIC_TRACK_MAX_AGE_SECONDS", 60.0) if is_static else profile["TRACK_MAX_AGE_SECONDS"]
            
            if current_time - tracker.last_seen > max_age:
                stale_tracker_ids.append(tracker_id)
    
    for tracker_id in stale_tracker_ids:
        del KALMAN_TRACKERS[tracker_id]
    
    # === PRESERVED: BOUNDING BOXES AND VELOCITY VECTORS ===
    bbox_traces, object_list_for_table = [], []
    kalman_prediction_traces = []
    
    for tracker in KALMAN_TRACKERS.values():
        current_pos = tracker.position
        box_size = 0.5
        
        bbox_trace = create_kalman_bounding_box_trace(current_pos, box_size, tracker.color)
        if bbox_trace:
            bbox_traces.append(bbox_trace)
        
        velocity_trace = create_velocity_vector_trace(current_pos, tracker.velocity, tracker.color)
        if velocity_trace:
            kalman_prediction_traces.append(velocity_trace)
        
        status = "Confirmed" if tracker.is_confirmed else "Tracking..."
        velocity_magnitude = np.linalg.norm(tracker.velocity)
        range_from_origin = np.linalg.norm(current_pos)
        
        object_list_for_table.append({
            "id": f"{tracker.label} #{tracker.id}",
            "status": status,
            "points": tracker.hits,
            "range": f"{range_from_origin:.2f}",
            "velocity": f"{velocity_magnitude:.2f}"
        })
    
    # === PRESERVED: PERFORMANCE-OPTIMIZED VISUALIZATION ===
    df_sampled_plot = memory_efficient_point_sampling(df_plot, max_points=max_display_points)
    plot_traces = []

    # Check what elements to display
    show_points = 'POINTS' in filter_options
    show_surfaces = 'SURFACE' in filter_options and not skip_dense_cloud
    show_grid = 'GRID' in filter_options
    show_doppler_effects = 'DOPPLER' in filter_options

    # Determine colorbar positioning
    if show_points and show_surfaces:
        points_colorbar_x = 1.02
        surface_colorbar_x = 1.12
    elif show_points and not show_surfaces:
        points_colorbar_x = 1.02
        surface_colorbar_x = None
    elif not show_points and show_surfaces:
        points_colorbar_x = None
        surface_colorbar_x = 1.02
    else:
        points_colorbar_x = None
        surface_colorbar_x = None

    # Initialize drone_points as an empty DataFrame to avoid UnboundLocalError
    drone_points = pd.DataFrame()

    # === ENHANCED POINT VISUALIZATION WITH DRONE HIGHLIGHTING ===
    if not df_sampled_plot.empty and show_points:
        # Separate drone points from regular points
        has_micro_doppler = 'micro_doppler_type' in df_sampled_plot.columns
        drone_points = df_sampled_plot[df_sampled_plot['micro_doppler_type'] == 'DRONE_CONFIRMED'] if has_micro_doppler else pd.DataFrame()
        regular_points = df_sampled_plot[df_sampled_plot.get('micro_doppler_type', '') != 'DRONE_CONFIRMED'] if has_micro_doppler else df_sampled_plot
        
        # Color mapping for regular points
        color_map = {'prop': 'velocity', 'scale': 'RdBu', 'label': 'Velocity', 'min': -2, 'max': 2}
        if display_mode == 'range': 
            color_map = {'prop': 'range', 'scale': 'Viridis', 'label': 'Range', 'min': 0, 'max': CONFIG['signal_processing']['MAX_RANGE_DISPLAY']}
        elif display_mode == 'signal_strength': 
            color_map = {'prop': 'snr', 'scale': 'Plasma', 'label': 'SNR', 'min': -90, 'max': -40}
        
        # Regular points
        if not regular_points.empty:
            plot_traces.append(go.Scatter3d(
                x=regular_points['x'], y=regular_points['y'], z=regular_points['z'], 
                mode='markers', name='Radar Points', showlegend=False,
                hovertemplate='Range: %{customdata[0]:.2f}m<br>Velocity: %{customdata[1]:.2f}m/s<br>SNR: %{customdata[2]:.1f}dB<extra></extra>',
                customdata=np.column_stack([regular_points['range'], regular_points['velocity'], regular_points['snr']]),
                marker=dict(
                    size=point_size,
                    color=regular_points[color_map['prop']], 
                    colorscale=color_map['scale'], 
                    cmin=color_map['min'], 
                    cmax=color_map['max'],
                    opacity=0.8,
                    showscale=True if points_colorbar_x else False,
                    colorbar=dict(
                        title=dict(text=color_map['label'], side="right"),
                        x=points_colorbar_x,
                        len=0.6,
                        thickness=8,
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0
                    ) if points_colorbar_x else None
                )
            ))
        
        # HIGHLIGHTED DRONE POINTS
        if not drone_points.empty:
            plot_traces.append(go.Scatter3d(
                x=drone_points['x'], y=drone_points['y'], z=drone_points['z'], 
                mode='markers', name='CONFIRMED DRONES', showlegend=True,
                hovertemplate='🚁 DRONE DETECTED<br>Range: %{customdata[0]:.2f}m<br>Velocity: %{customdata[1]:.2f}m/s<br>Blade Freq: %{customdata[2]:.1f}Hz<br>Threat: %{customdata[3]}<extra></extra>',
                customdata=np.column_stack([
                    drone_points['range'], 
                    drone_points['velocity'], 
                    drone_points.get('blade_frequency', 0),
                    drone_points.get('threat_level', 'UNKNOWN')
                ]),
                marker=dict(
                    size=point_size * 2,  # Larger for visibility
                    color='red',  # Always red for drones
                    symbol='diamond',  # Different shape
                    opacity=1.0,  # Fully opaque
                    line=dict(color='yellow', width=2)  # Yellow outline
                )
            ))

    # === ENHANCED: MICRO-DOPPLER SIGNATURE VISUALIZATION ===
    if not drone_points.empty and show_doppler_effects:
        # Add pulsing rings around confirmed drones
        for _, drone in drone_points.iterrows():
            blade_freq = drone.get('blade_frequency', 50)
            
            # Create pulsing ring based on blade frequency
            theta = np.linspace(0, 2*np.pi, 50)
            ring_radius = 0.5 + 0.3 * np.sin(time.time() * blade_freq / 10)  # Pulse at blade frequency
            
            ring_x = drone['x'] + ring_radius * np.cos(theta)
            ring_y = drone['y'] + ring_radius * np.sin(theta)
            ring_z = np.full_like(theta, drone['z'])
            
            plot_traces.append(go.Scatter3d(
                x=ring_x, y=ring_y, z=ring_z,
                mode='lines',
                line=dict(color='yellow', width=4),
                name=f'Blade Flash {blade_freq:.0f}Hz',
                showlegend=False,
                hovertemplate=f'Micro-Doppler Ring<br>Blade Frequency: {blade_freq:.1f}Hz<extra></extra>'
            ))
            
            # Add frequency text label
            plot_traces.append(go.Scatter3d(
                x=[drone['x']], y=[drone['y']], z=[drone['z'] + 0.5],
                mode='text',
                text=[f'{blade_freq:.0f}Hz'],
                textfont=dict(color='yellow', size=12),
                showlegend=False,
                hoverinfo='none'
            ))

    # === PRESERVED: Surface reconstruction with semantic classification ===
    dense_point_count = 0
    if show_surfaces:
        # Extract semantic objects for voxel classification - FIXED with stricter criteria
        semantic_objects = []
        if not df_recent.empty and len(df_recent) > 10:  # Increased minimum requirement
            # More conservative clustering to reduce false positives
            clusters = detect_objects_dbscan(df_recent, eps=0.5, min_samples=8)  # UPGRADED: Stricter parameters for 500 MHz
            for cluster_df in clusters:
                if len(cluster_df) >= 8:  # Higher minimum
                    semantic_obj = classify_semantic_object(cluster_df, environment_mode)
                    if semantic_obj and semantic_obj['confidence'] > 0.8:  # Much higher confidence threshold
                        semantic_objects.append(semantic_obj)
        
        # Generate enhanced voxel grid with semantic information
        dense_points = adaptive_dense_cloud_generation(
            df_sampled_plot,
            actual_performance_mode,
            semantic_objects,
            voxel_resolution=voxel_resolution,
            voxel_size=voxel_size
        )
        
        # Process voxel traces
        if dense_points:
            if len(dense_points) > 0 and hasattr(dense_points[0], 'type'):
                # New voxel grid system - dense_points is already a list of Plotly traces
                plot_traces.extend(dense_points)
                dense_point_count = len(dense_points)
                logger.info(f"Added {len(dense_points)} voxel traces to visualization")
            else:
                logger.warning("Dense points returned but format not recognized")
        else:
            logger.info("No dense points generated")

    max_range = CONFIG['signal_processing']['MAX_RANGE_DISPLAY']

    # === PRESERVED: RADAR INFRASTRUCTURE ===
    plot_traces.extend(create_radar_cone_traces_with_opacity(max_range, CONFIG['scanning']['AZ_FOV'], cone_opacity))
    plot_traces.append(go.Scatter3d(
        x=[0, max_range * np.sin(np.radians(current_az))], 
        y=[0, max_range * np.cos(np.radians(current_az))], 
        z=[0, 0], 
        mode='lines', line=dict(color='#00FF00', width=5), 
        hoverinfo='none', name='Current Beam', showlegend=False
    ))
    plot_traces.extend(create_range_rings(max_range))
    plot_traces.extend(bbox_traces)
    plot_traces.extend(kalman_prediction_traces)

    # === ENHANCED DUAL LEGEND SYSTEM ===
    # Add semantic legend if voxels are displayed (positioned on RIGHT)
    # 1. Add Semantic Voxel Legend (RIGHT side)
    if show_surfaces:
        plot_traces.extend(create_semantic_legend())

    # 2. Add Micro-Doppler Legend (RIGHT side, stacked below)
    plot_traces.extend(create_micro_doppler_legend())
    
    # 3. Add Cluster Box Legend (LEFT side)
    if any(KALMAN_TRACKERS):
        plot_traces.extend(create_cluster_box_legend())

    # === ENHANCED HEARTBEAT WITH DRONE STATUS ===
    heartbeat_age = current_time - last_heartbeat if last_heartbeat > 0 else float('inf')
    drone_age = current_time - last_drone_detection if last_drone_detection > 0 else float('inf')
    
    if heartbeat_age < 5:
        pulse_phase = (current_time * 2) % (2 * np.pi)
        base_radius = 1.5
        pulse_radius = base_radius + 0.8 * np.sin(pulse_phase)
        pulse_intensity = 0.6 + 0.4 * np.sin(pulse_phase)
        pulse_color = f'rgba(40, 167, 69, {pulse_intensity})'
        
        theta = np.linspace(0, 2*np.pi, 60)
        pulse_x = pulse_radius * np.cos(theta)
        pulse_y = pulse_radius * np.sin(theta)
        pulse_z = np.zeros_like(theta) + 0.2
        
        plot_traces.append(go.Scatter3d(
            x=pulse_x, y=pulse_y, z=pulse_z,
            mode='lines+markers', name='System Heartbeat',
            line=dict(color=pulse_color, width=6),
            marker=dict(size=3, color=pulse_color),
            hovertemplate=f'System Health: GOOD<br>Last Heartbeat: {heartbeat_age:.1f}s ago<extra></extra>',
            showlegend=False
        ))
        
        # Enhanced status with drone detection info
        if drone_age < 10:
            status_text = f'♥ DRONE THREAT DETECTED ({drone_age:.0f}s ago)'
            status_color = '#dc3545'  # Red for drone threat
        else:
            status_text = f'♥ MICRO-DOPPLER ACTIVE: {actual_performance_mode.upper()}'
            status_color = '#28a745'  # Green for normal
            
        plot_traces.append(go.Scatter3d(
            x=[0], y=[-1], z=[1], mode='text',
            text=[status_text],
            textfont=dict(color=status_color, size=14),
            hovertemplate=f'Performance: {actual_performance_mode}<br>Kalman Trackers: {len(KALMAN_TRACKERS)}<br>Dense Points: {dense_point_count}<br>Drone Detections: {drone_detections}<br>Heartbeat: {heartbeat_age:.1f}s ago<extra></extra>',
            showlegend=False, name=''
        ))
    elif heartbeat_age < 10:
        plot_traces.append(go.Scatter3d(
            x=[0], y=[-1], z=[1], mode='text',
            text=[f'⚠ RADAR DEGRADED'],
            textfont=dict(color='#ffc107', size=14),
            hoverinfo='none', showlegend=False, name=''
        ))
    else:
        plot_traces.append(go.Scatter3d(
            x=[0], y=[-1], z=[1], mode='text',
            text=[f'💀 RADAR LOST'],
            textfont=dict(color='#dc3545', size=16),
            hoverinfo='none', showlegend=False, name=''
        ))
    
    # === PRESERVED: FIGURE ASSEMBLY ===
    fig = go.Figure(data=plot_traces)
    
    heartbeat_status = "HEALTHY" if heartbeat_age < 5 else "WEAK" if heartbeat_age < 10 else "LOST"
    heartbeat_color = "#28a745" if heartbeat_age < 5 else "#ffc107" if heartbeat_age < 10 else "#dc3545"
    
    az_fov_deg = CONFIG['scanning']['AZ_FOV']
    
    # PRESERVED: CN0566 Drone Radar System info display
    info_text = (
        f"<b>CN0566 Drone Radar System</b><br>"
        f"Max Range: {CONFIG['signal_processing']['MAX_RANGE_DISPLAY']:.1f}m<br>"
        f"Azimuth FoV: {abs(az_fov_deg[0]) + abs(az_fov_deg[1])}° ({az_fov_deg[0]}° to {az_fov_deg[1]}°)<br>"
        f"Elevation FoV: {CONFIG['scanning']['VERTICAL_BEAMWIDTH']}°<br>"
        f"RF Frequency: {CONFIG['hardware']['OUTPUT_FREQ']/1e9:.2f} GHz<br>"
        f"Array Elements: {CONFIG['hardware']['NUM_AZ_ELEMENTS']}<br>"
        f"<span style='color:#1DB954'><b>Tracked Objects: {len(KALMAN_TRACKERS)}</b></span><br>"
        f"<span style='color:#00CED1'><b>Surface Voxels: {dense_point_count}</b></span><br>"
        f"<span style='color:#dc3545'><b>Drone Detections: {drone_detections}</b></span><br>"
        f"<span style='color:#ffc107'><b>Mode: {actual_performance_mode.upper()}</b></span><br>"
        f"<span style='color:{heartbeat_color}'><b>Pi Status: {heartbeat_status}</b></span>"
    )
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000000", margin=dict(l=0, r=0, b=0, t=0),
        uirevision='pointcloud', showlegend=False,
        scene=dict(
            xaxis=dict(visible=False, range=[-max_range, max_range]),
            yaxis=dict(visible=False, range=[-5, max_range]),
            zaxis=dict(visible=False, range=[-max_range/2, max_range/2]),
            bgcolor="#000000", aspectratio=dict(x=2, y=2, z=1)
        ),
        annotations=[
            dict(
                text=info_text, align='left', showarrow=False,
                xref='paper', yref='paper', x=0.02, y=0.02,
                bordercolor='rgba(255, 255, 255, 0.5)', borderwidth=1,
                bgcolor='rgba(40, 40, 40, 0.8)', font=dict(color='white', size=12)
            )
        ]
    )
    
    # === PRESERVED: ENHANCED METRICS CALCULATION ===
    loop_time_ms = (time.time() - start_time) * 1000
    try:
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_percent = (cpu_freq.current / cpu_freq.max * 100) if cpu_freq else 100
        performance_color = "#28a745" if actual_performance_mode == 'full' else "#ffc107" if actual_performance_mode == 'balanced' else "#dc3545"
        
        system_metrics_children = html.Div(className="metric-grid", children=[
            html.Div(className="metric-item", children=[
                html.Div(f"{loop_time_ms:.1f} ms", className="metric-value"), 
                html.Div("Loop Time", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div(f"{cpu_percent:.1f}%", className="metric-value"), 
                html.Div("CPU Usage", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div(f"{memory_info.percent:.1f}%", className="metric-value"), 
                html.Div("Memory", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div(f"{cpu_freq_percent:.0f}%", className="metric-value"), 
                html.Div("CPU Freq", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div(actual_performance_mode.upper(), className="metric-value", 
                         style={'color': performance_color}), 
                html.Div("Mode", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div(f"{drone_detections}", className="metric-value", 
                         style={'color': '#dc3545' if drone_detections > 0 else '#28a745'}), 
                html.Div("Drones", className="metric-label")
            ]),
        ])
    except:
        system_metrics_children = html.Div(className="metric-grid", children=[
            html.Div(className="metric-item", children=[
                html.Div(f"{loop_time_ms:.1f} ms", className="metric-value"), 
                html.Div("Loop Time", className="metric-label")
            ]),
            html.Div(className="metric-item", children=[
                html.Div("N/A", className="metric-value"), 
                html.Div("System", className="metric-label")
            ]),
        ])
    
    point_count_text = [html.Div(f"{len(df_plot)}", className="metric-value"), html.Div("Displayed", className="metric-label")]
    current_az_text = [html.Div(f"{current_az:.1f}°", className="metric-value"), html.Div("Azimuth", className="metric-label")]
    objects_detected_text = [html.Div(f"{len(KALMAN_TRACKERS)}", className="metric-value"), html.Div("Tracked Objects", className="metric-label")]
    scan_count_text = [html.Div(f"{scan_count}", className="metric-value"), html.Div("Scans", className="metric-label")]

    return fig, point_count_text, current_az_text, objects_detected_text, scan_count_text, object_list_for_table, system_metrics_children

@app.callback(
    Output("waterfall-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_waterfall_display(n):
    with RADAR_DATA_LOCK:
        waterfall_data = list(RADAR_DATA['waterfall'])
        data_ready = RADAR_DATA['data_is_ready']

    if not data_ready or not waterfall_data:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#282828")

    waterfall_array = np.array(waterfall_data)

    fig = go.Figure(data=go.Heatmap(
        z=waterfall_array,
        x=range_m_axis[:waterfall_array.shape[1]] if waterfall_array.size > 0 else [],
        y=list(range(len(waterfall_data))),
        colorscale='Plasma',
        hovertemplate='Range: %{x:.1f}m<br>Scan: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#282828",
        plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=20),
        xaxis_title="Range (m)",
        yaxis_title="Scan Number"
    )

    return fig

@app.callback(
    Output("beampattern-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_beampattern_display(n):
    with RADAR_DATA_LOCK:
        beampattern = RADAR_DATA['beampattern']
        current_az = RADAR_DATA['current_az']
        data_ready = RADAR_DATA['data_is_ready']

    if not data_ready or beampattern is None:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#282828")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=az_angles_beampattern,
        y=beampattern,
        mode='lines',
        name='Beampattern',
        line=dict(color='#1DB954', width=2)
    ))

    fig.add_vline(x=current_az, line_dash="dash", line_color="#ff4444",
                  annotation_text=f"Current: {current_az:.1f}°")

    fig.update_layout(
        template="plotly_dark",
        uirevision='beampattern',
        paper_bgcolor="#282828",
        plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=20),
        xaxis_title="Azimuth Angle (degrees)",
        yaxis_title="Gain (dB)",
        xaxis=dict(range=[-60, 60]),
        yaxis=dict(range=[-40, 5])
    )

    return fig

@app.callback(
    Output('tabs-content', 'children'),
    [Input('diagnostic-tabs', 'value')]
)
def render_tab_content(active_tab):
    """Render content based on active diagnostic tab."""
    if active_tab == 'tab-doppler':
        return dcc.Graph(id="range-doppler-plot", style={'height': '280px'})
    # --- ADD THESE TWO NEW CONDITIONS ---
    elif active_tab == 'tab-elevation':
        return dcc.Graph(id="elevation-heatmap-plot", style={'height': '280px'})
    elif active_tab == 'tab-top-down':
        return dcc.Graph(id="top-down-heatmap-plot", style={'height': '280px'})
    # --- END OF ADDITION ---
    elif active_tab == 'tab-micro-doppler':
        return dcc.Graph(id="micro-doppler-plot", style={'height': '280px'})
    elif active_tab == 'tab-waterfall':
        return dcc.Graph(id="waterfall-plot", style={'height': '280px'})
    elif active_tab == 'tab-beam':
        return dcc.Graph(id="beampattern-plot", style={'height': '280px'})
    return html.Div("Select a tab")

@app.callback(
    Output("elevation-heatmap-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_elevation_heatmap(n):
    """
    Creates a 2D heatmap of Azimuth vs. Elevation.
    This shows the vertical and horizontal distribution of radar detections.
    """
    with RADAR_DATA_LOCK:
        all_points_list = [p for d in RADAR_DATA['persistent_points'].values() for p in d]
        data_ready = RADAR_DATA['data_is_ready']

    if not data_ready or not all_points_list:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#282828",
                                       annotations=[dict(text="Awaiting Detections", showarrow=False)])

    df = pd.DataFrame(all_points_list)

    # Calculate elevation from XYZ coordinates
    df['elevation'] = np.degrees(np.arctan2(df['z'], np.sqrt(df['x']**2 + df['y']**2)))

    fig = go.Figure(data=go.Histogram2d(
        x=df['azimuth'],
        y=df['elevation'],
        colorscale='Plasma',
        z=df['snr'],  # Weight bins by signal strength
        histfunc='avg',
        xbins=dict(start=-60, end=60, size=5),  # 5-degree azimuth bins
        ybins=dict(start=-30, end=30, size=5),  # 5-degree elevation bins
        colorbar=dict(title="Avg. SNR (dB)")
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#282828",
        plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=40),
        title="Elevation vs. Azimuth FoV",
        xaxis_title="Azimuth (degrees)",
        yaxis_title="Elevation (degrees)"
    )
    return fig


@app.callback(
    Output("top-down-heatmap-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_top_down_heatmap(n):
    """
    Creates a top-down 2D heatmap (similar to a Plan Position Indicator).
    This shows detections on an X/Y grid from a bird's-eye view.
    """
    with RADAR_DATA_LOCK:
        all_points_list = [p for d in RADAR_DATA['persistent_points'].values() for p in d]
        max_range = CONFIG['signal_processing']['MAX_RANGE_DISPLAY']
        data_ready = RADAR_DATA['data_is_ready']

    if not data_ready or not all_points_list:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#282828",
                                       annotations=[dict(text="Awaiting Detections", showarrow=False)])

    df = pd.DataFrame(all_points_list)

    fig = go.Figure(data=go.Histogram2d(
        x=df['x'],
        y=df['y'],
        colorscale='Viridis',
        z=df['snr'],
        histfunc='avg',
        xbins=dict(start=-max_range, end=max_range, size=1), # 1-meter X bins
        ybins=dict(start=0, end=max_range, size=1),         # 1-meter Y bins
        colorbar=dict(title="Avg. SNR (dB)")
    ))

    # Add the radar cone outline for context
    az_fov_deg = CONFIG['scanning']['AZ_FOV']
    az_rad = np.radians(np.linspace(az_fov_deg[0], az_fov_deg[1], 50))
    fig.add_trace(go.Scatter(
        x=max_range * np.sin(az_rad),
        y=max_range * np.cos(az_rad),
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.5)', dash='dot'),
        hoverinfo='none'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#282828",
        plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=40),
        title="Top-Down View (X-Y Plane)",
        xaxis_title="X-Axis (meters)",
        yaxis_title="Y-Axis (meters)",
        showlegend=False
    )
    # Ensure the plot is not stretched
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

@app.callback(
    Output("range-doppler-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_range_doppler_display(n):
    """Update the range-doppler heatmap display."""
    with RADAR_DATA_LOCK:
        rd_map = RADAR_DATA['range_doppler_map']
        data_ready = RADAR_DATA['data_is_ready']

    if not data_ready or rd_map is None:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#282828")

    fig = go.Figure(data=go.Heatmap(
        z=rd_map,
        x=range_m_axis[:rd_map.shape[1]] if rd_map.size > 0 else [],
        y=velocity_axis[:rd_map.shape[0]] if rd_map.size > 0 else [],
        colorscale='Plasma',
        colorbar=dict(title="Power (dB)"),
        hovertemplate='Range: %{x:.1f}m<br>Velocity: %{y:.1f}m/s<br>Power: %{z:.1f}dB<extra></extra>'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#282828",
        plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=20),
        xaxis_title="Range (m)",
        yaxis_title="Velocity (m/s)"
    )

    return fig

@app.callback(
    Output('selection-info-card', 'children'),
    Input('point-cloud-3d', 'clickData'),
    prevent_initial_call=True
)
def update_selection_info_enhanced(click_data):
    """FIXED: Enhanced callback with proper customdata extraction."""
    if click_data is None or not click_data['points']:
        return [
            html.H2("Point Information"),
            html.P("Click on a point to see detailed information.",
                   style={'color': '#B3B3B3', 'font-style': 'italic'})
        ]

    point = click_data['points'][0]
    x, y, z = point.get('x', 0), point.get('y', 0), point.get('z', 0)

    # FIXED: Proper customdata extraction
    if 'customdata' in point and point['customdata'] is not None:
        try:
            customdata = point['customdata']
            if isinstance(customdata, (list, tuple)) and len(customdata) >= 3:
                range_3d_val = float(customdata[0])
                velocity = f"{float(customdata[1]):.2f} m/s"
                snr = f"{float(customdata[2]):.1f} dB"
                snr_val = float(customdata[2])
            else:
                raise ValueError("Invalid customdata format")
        except (ValueError, TypeError, IndexError):
            range_3d_val = np.sqrt(x**2 + y**2 + z**2)
            velocity = "N/A"
            snr = "N/A"
            snr_val = -120
    else:
        range_3d_val = np.sqrt(x**2 + y**2 + z**2)
        velocity = "N/A"
        snr = "N/A"
        snr_val = -120

    range_3d = f"{range_3d_val:.2f} m"
    range_2d = f"{np.sqrt(x**2 + y**2):.2f} m"
    azimuth = f"{np.degrees(np.arctan2(x, y)):.1f}°"
    elevation = f"{np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))):.1f}°"

    point_type = "Strong" if snr_val > -75 else "Medium" if snr_val > -95 else "Weak"

    return [
        html.H2("Selected Point Details"),
        html.Div(className="selection-details", children=[
            html.H4("Spatial Information", style={'margin-bottom': '8px', 'color': '#1DB954'}),
            html.Div(className="coordinate-display", children=[
                html.Div([html.Strong("X:"), html.Br(), f"{x:.2f} m"], className="coordinate-item"),
                html.Div([html.Strong("Y:"), html.Br(), f"{y:.2f} m"], className="coordinate-item"),
                html.Div([html.Strong("Z:"), html.Br(), f"{z:.2f} m"], className="coordinate-item"),
            ]),
            html.H4("Orientation & Range", style={'margin': '12px 0 8px 0', 'color': '#1DB954'}),
            html.Div(className="metric-grid", children=[
                html.Div([html.Div(range_3d, className="metric-value"), html.Div("3D Range", className="metric-label")], className="metric-item"),
                html.Div([html.Div(azimuth, className="metric-value"), html.Div("Azimuth", className="metric-label")], className="metric-item"),
                html.Div([html.Div(elevation, className="metric-value"), html.Div("Elevation", className="metric-label")], className="metric-item"),
                html.Div([html.Div(range_2d, className="metric-value"), html.Div("Ground Range", className="metric-label")], className="metric-item"),
            ]),
            html.H4("Signal Properties", style={'margin': '12px 0 8px 0', 'color': '#1DB954'}),
            html.Div(className="metric-grid", children=[
                html.Div([html.Div(velocity, className="metric-value"), html.Div("Velocity", className="metric-label")], className="metric-item"),
                html.Div([html.Div(snr, className="metric-value"), html.Div("SNR", className="metric-label")], className="metric-item"),
                html.Div([html.Div(point_type, className="metric-value"), html.Div("Type", className="metric-label")], className="metric-item"),
            ])
        ])
    ]


@app.callback(
    [Output("status-indicator", "className"),
     Output("status-text", "children"),
     Output("progress-fill", "style"),
     Output("progress-text", "children")],
    [Input("fast-update-interval", "n_intervals")]
)
def update_status_and_progress(n):
    """UPGRADED to provide a visual heartbeat pulse on the status indicator."""
    with RADAR_DATA_LOCK:
        error = RADAR_DATA.get('error')
        scan_progress = RADAR_DATA.get('scan_progress', 0)
        data_ready = RADAR_DATA.get('data_is_ready', False)
        last_update = RADAR_DATA.get('last_update', 0)
        last_heartbeat = RADAR_DATA.get('last_heartbeat_time', 0)

    # Heartbeat Check
    time_since_heartbeat = time.time() - last_heartbeat
    if time_since_heartbeat > 5.0:
        status_class = "status-indicator status-error" # Solid Red
        status_text = f"Pi Connection Lost ({time_since_heartbeat:.0f}s)"
        
    # Other Status Checks
    elif error:
        status_class = "status-indicator status-error" # Solid Red
        status_text = f"Error: {error[:50]}..."
    elif not data_ready:
        status_class = "status-indicator status-warning" # Solid Yellow
        status_text = "Initializing..."
    elif time.time() - last_update > 5:
        status_class = "status-indicator status-warning" # Solid Yellow
        status_text = "Data stale"
    else:
        # --- THE CHANGE ---
        # If everything is OK, use the pulsing heartbeat animation class
        status_class = "status-indicator status-ok status-heartbeat" # Pulsing Green
        status_text = "Operating normally"
        # ------------------

    # Progress bar logic
    progress_percent = int(scan_progress * 100)
    progress_style = {"width": f"{progress_percent}%"}
    progress_text = f"{progress_percent}%"

    return status_class, status_text, progress_style, progress_text

@app.callback(
    [Output('sidebar', 'style'),
     Output('main-view', 'style'),
     Output('sidebar-toggle-btn', 'children'),
     Output('sidebar-state-store', 'data')],
    [Input('sidebar-toggle-btn', 'n_clicks')],
    [State('sidebar-state-store', 'data')],
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, stored_data):
    is_expanded = stored_data.get('expanded', True)
    new_expanded_state = not is_expanded

    if new_expanded_state:
        # Expand it
        sidebar_style = {'minWidth': '480px', 'width': '480px', 'padding': '20px'}
        main_view_style = {'width': 'calc(100% - 480px)'}
        button_text = '‹'

    else:
        sidebar_style = {'minWidth': '0px', 'width': '0px', 'padding': '0'}
        main_view_style = {'width': '100%'}
        button_text = '›'

    new_data = {'expanded': new_expanded_state}
    return sidebar_style, main_view_style, button_text, new_data
    
@app.callback(
    Output('environment-mode-radio', 'value'),
    [Input('fast-update-interval', 'n_intervals'),
     Input('environment-mode-radio', 'value')],
    [State('environment-mode-radio', 'value')],
    prevent_initial_call=True
)
def update_environment_mode(n_intervals, radio_value, current_state):
    """
    FIXED: Handle automatic and manual environment mode switching.
    """
    ctx = callback_context
    if not ctx.triggered:
        return current_state
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'environment-mode-radio':
        # User manually changed the radio button
        with RADAR_DATA_LOCK:
            RADAR_DATA['active_profile_name'] = radio_value
            RADAR_DATA['manual_environment_override'] = True
        return radio_value
    
    elif trigger_id == 'fast-update-interval':
        # Automatic update from receiver thread
        with RADAR_DATA_LOCK:
            auto_mode = RADAR_DATA.get('active_profile_name', 'indoor')
            manual_override = RADAR_DATA.get('manual_environment_override', False)
        
        # Only update automatically if user hasn't manually overridden
        if not manual_override:
            return auto_mode
        else:
            return current_state
    
    return current_state

@app.callback(
    Output('performance-mode-dropdown', 'value'),
    [Input('fast-update-interval', 'n_intervals'),
     Input('performance-mode-dropdown', 'value')],
    [State('performance-mode-dropdown', 'value')],
    prevent_initial_call=True
)
def update_performance_mode(n_intervals, dropdown_value, current_state):
    """
    COMPLETE: Handle automatic and manual performance mode switching with full auto-detection
    - Preserves all original auto-detection logic
    - Monitors CPU, memory, and system performance
    - Automatically adjusts based on system load
    - Supports manual override
    """
    ctx = callback_context
    if not ctx.triggered:
        return current_state
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'performance-mode-dropdown':
        # User manually changed the mode - store the override
        if not hasattr(update_performance_mode, 'manual_override'):
            update_performance_mode.manual_override = {}
        
        update_performance_mode.manual_override['active'] = True
        update_performance_mode.manual_override['value'] = dropdown_value
        update_performance_mode.manual_override['timestamp'] = time.time()
        
        return dropdown_value
    
    elif trigger_id == 'fast-update-interval':
        # Automatic update based on system performance
        
        # Check if user has manually overridden recently (within 30 seconds)
        if hasattr(update_performance_mode, 'manual_override'):
            if (update_performance_mode.manual_override.get('active', False) and
                time.time() - update_performance_mode.manual_override.get('timestamp', 0) < 30):
                return update_performance_mode.manual_override['value']
        
        # FULL AUTO-DETECTION LOGIC
        if dropdown_value == 'auto':
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                # Get CPU frequency if available
                try:
                    cpu_freq = psutil.cpu_freq()
                    cpu_freq_percent = (cpu_freq.current / cpu_freq.max * 100) if cpu_freq else 100
                except:
                    cpu_freq_percent = 100
                
                # Get disk I/O if available
                try:
                    disk_io = psutil.disk_io_counters()
                    disk_busy = disk_io.read_time + disk_io.write_time if disk_io else 0
                except:
                    disk_busy = 0
                
                # Calculate performance score
                performance_score = 100
                
                # CPU load penalty
                if cpu_percent > 80:
                    performance_score -= 40
                elif cpu_percent > 60:
                    performance_score -= 20
                elif cpu_percent > 40:
                    performance_score -= 10
                
                # Memory usage penalty
                if memory_info.percent > 85:
                    performance_score -= 30
                elif memory_info.percent > 70:
                    performance_score -= 15
                elif memory_info.percent > 50:
                    performance_score -= 5
                
                # CPU frequency penalty
                if cpu_freq_percent < 70:
                    performance_score -= 20
                elif cpu_freq_percent < 85:
                    performance_score -= 10
                
                # Thermal throttling detection (rough estimate)
                if cpu_percent > 70 and cpu_freq_percent < 80:
                    performance_score -= 25  # Likely thermal throttling
                
                # Check available memory
                available_gb = memory_info.available / (1024**3)
                if available_gb < 1.0:
                    performance_score -= 35
                elif available_gb < 2.0:
                    performance_score -= 20
                elif available_gb < 4.0:
                    performance_score -= 10
                
                # Network performance factor
                network_latency = monitor_network_performance()
                if network_latency:
                    if network_latency > 20:
                        performance_score -= 15
                    elif network_latency > 10:
                        performance_score -= 8
                
                # Time-based adaptive logic
                current_time = time.time()
                if not hasattr(update_performance_mode, 'performance_history'):
                    update_performance_mode.performance_history = deque(maxlen=10)
                
                update_performance_mode.performance_history.append({
                    'time': current_time,
                    'cpu': cpu_percent,
                    'memory': memory_info.percent,
                    'score': performance_score
                })
                
                # Smooth performance transitions
                if len(update_performance_mode.performance_history) >= 3:
                    recent_scores = [h['score'] for h in list(update_performance_mode.performance_history)[-3:]]
                    avg_score = np.mean(recent_scores)
                    score_std = np.std(recent_scores)
                    
                    # Prefer stability - don't change modes too frequently
                    if score_std > 15:  # High variation - use conservative estimate
                        performance_score = min(recent_scores)
                
                # Determine performance mode based on score
                if performance_score >= 70:
                    auto_mode = 'full'
                elif performance_score >= 40:
                    auto_mode = 'balanced'
                else:
                    auto_mode = 'minimal'
                
                # Add hysteresis to prevent rapid switching
                if hasattr(update_performance_mode, 'last_auto_mode'):
                    last_mode = update_performance_mode.last_auto_mode
                    
                    # Only switch if there's a significant difference
                    if last_mode == 'full' and auto_mode == 'balanced' and performance_score > 60:
                        auto_mode = 'full'  # Stay in full mode
                    elif last_mode == 'minimal' and auto_mode == 'balanced' and performance_score < 50:
                        auto_mode = 'minimal'  # Stay in minimal mode
                
                update_performance_mode.last_auto_mode = auto_mode
                
                # Log performance decision
                logger.info(f"Auto performance mode: {auto_mode} (score: {performance_score:.1f}, "
                           f"CPU: {cpu_percent:.1f}%, Memory: {memory_info.percent:.1f}%)")
                
                return auto_mode
                
            except Exception as e:
                logger.error(f"Performance mode auto-detection failed: {e}")
                return 'balanced'  # Safe fallback
        
        else:
            # User has selected a specific mode
            return dropdown_value
    
    return current_state

@app.callback(
    Output('point-cloud-3d', 'figure', allow_duplicate=True),
    Input('reset-camera-btn', 'n_clicks'),
    State('point-cloud-3d', 'figure'),
    prevent_initial_call=True
)
def reset_camera_view(n_clicks, current_figure):
    """
    Reset camera to boresight view when button is clicked.
    """
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    if current_figure is None:
        raise PreventUpdate
    
    # Create a copy and only modify the camera
    import copy
    updated_figure = copy.deepcopy(current_figure)
    
    # Ensure layout structure exists
    if 'layout' not in updated_figure:
        updated_figure['layout'] = {}
    
    if 'scene' not in updated_figure['layout']:
        updated_figure['layout']['scene'] = {}
    
    # Set camera to boresight view
    updated_figure['layout']['scene']['camera'] = {
        'eye': {'x': 0, 'y': -1.8, 'z': 0.8},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'up': {'x': 0, 'y': 0, 'z': 1}
    }
    
    # Use a unique uirevision to force the camera update
    updated_figure['layout']['uirevision'] = f'camera-reset-{n_clicks}'
    
    return updated_figure

@app.callback(
    Output("micro-doppler-plot", "figure"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_micro_doppler_display(n):
    """
    FIXED: Use the same data as main display for consistency
    """
    with RADAR_DATA_LOCK:
        drone_detections = RADAR_DATA.get('drone_detections', 0)
        last_drone_detection = RADAR_DATA.get('last_drone_detection', 0)
    
    current_time = time.time()
    time_since_detection = current_time - last_drone_detection if last_drone_detection > 0 else float('inf')
    
    # Check if we have actual detections recently
    if drone_detections > 0 and time_since_detection < 10:
        # Get the same micro-Doppler history used in main processing
        if hasattr(process_live_data_with_micro_doppler, 'rd_history'):
            rd_history = list(process_live_data_with_micro_doppler.rd_history)
        else:
            return go.Figure().update_layout(
                template="plotly_dark", paper_bgcolor="#282828",
                annotations=[dict(text="No Active Micro-Doppler Detection", 
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(size=16, color="yellow"))]
            )
        
        if len(rd_history) < 20:
            return go.Figure().update_layout(
                template="plotly_dark", paper_bgcolor="#282828",
                annotations=[dict(text="Building Micro-Doppler History...", 
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(size=16, color="orange"))]
            )
        
        # Create spectrogram (simplified version)
        fig = go.Figure()
        
        # Add detection indicator
        fig.add_annotation(
            text=f"🚁 ACTIVE DETECTION<br>{drone_detections} signatures<br>{time_since_detection:.1f}s ago",
            xref="paper", yref="paper", x=0.5, y=0.8,
            showarrow=False, font=dict(size=14, color="red"),
            bgcolor="rgba(255,0,0,0.1)", bordercolor="red"
        )
        
        # Add frequency bands reference
        fig.add_trace(go.Scatter(
            x=[0, 10], y=[20, 20], mode='lines', name='Fan Range (5-30Hz)',
            line=dict(color='cyan', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 10], y=[100, 100], mode='lines', name='Drone Range (40-200Hz)',
            line=dict(color='red', dash='dash')
        ))
        
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="✓ MONITORING FOR MICRO-DOPPLER<br>No signatures detected", 
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="green")
        )
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#282828", plot_bgcolor="#282828",
        margin=dict(l=40, r=10, b=40, t=20),
        title="Micro-Doppler Status"
    )
    
    return fig

@app.callback(
    [Output("micro-doppler-status", "children"),
     Output("blade-frequency-display", "children"),
     Output("drone-confidence-display", "children"), 
     Output("signature-strength-display", "children"),
     Output("detection-type-display", "children")],
    [Input("fast-update-interval", "n_intervals")]
)
def update_micro_doppler_status(n):
    """Update micro-Doppler detection status and metrics."""
    with RADAR_DATA_LOCK:
        drone_detections = RADAR_DATA.get('drone_detections', 0)
        last_drone_detection = RADAR_DATA.get('last_drone_detection', 0)
    
    current_time = time.time()
    time_since_detection = current_time - last_drone_detection if last_drone_detection > 0 else float('inf')
    
    if drone_detections > 0 and time_since_detection < 5:
        # Active micro-Doppler detection
        status = html.Div([
            html.H4("🚁 ACTIVE MICRO-DOPPLER DETECTION", style={'color': '#dc3545', 'margin': '0'}),
            html.P(f"Drone signatures detected {time_since_detection:.1f}s ago", 
                   style={'color': '#ffc107', 'margin': '5px 0 0 0', 'font-size': '0.9rem'})
        ])
        
        # Mock data - in real implementation, get from actual detections
        blade_freq = [html.Div("85 Hz", className="metric-value"), html.Div("Blade Rate", className="metric-label")]
        confidence = [html.Div("94%", className="metric-value", style={'color': '#dc3545'}), html.Div("Confidence", className="metric-label")]
        strength = [html.Div("-45 dB", className="metric-value"), html.Div("Signature", className="metric-label")]
        drone_type = [html.Div("DJI-Type", className="metric-value"), html.Div("Classification", className="metric-label")]
        
    elif time_since_detection < 30:
        # Recent detection
        status = html.Div([
            html.H4("⚠ RECENT MICRO-DOPPLER ACTIVITY", style={'color': '#ffc107', 'margin': '0'}),
            html.P(f"Last detection: {time_since_detection:.0f}s ago", 
                   style={'color': '#B3B3B3', 'margin': '5px 0 0 0', 'font-size': '0.9rem'})
        ])
        
        blade_freq = [html.Div("--", className="metric-value"), html.Div("Blade Rate", className="metric-label")]
        confidence = [html.Div("--", className="metric-value"), html.Div("Confidence", className="metric-label")]
        strength = [html.Div("--", className="metric-value"), html.Div("Signature", className="metric-label")]
        drone_type = [html.Div("--", className="metric-value"), html.Div("Classification", className="metric-label")]
        
    else:
        # No detection
        status = html.Div([
            html.H4("✓ MICRO-DOPPLER MONITORING", style={'color': '#28a745', 'margin': '0'}),
            html.P("Scanning for blade flash signatures...", 
                   style={'color': '#B3B3B3', 'margin': '5px 0 0 0', 'font-size': '0.9rem'})
        ])
        
        blade_freq = [html.Div("--", className="metric-value"), html.Div("Blade Rate", className="metric-label")]
        confidence = [html.Div("--", className="metric-value"), html.Div("Confidence", className="metric-label")]
        strength = [html.Div("--", className="metric-value"), html.Div("Signature", className="metric-label")]
        drone_type = [html.Div("Clear", className="metric-value", style={'color': '#28a745'}), html.Div("Airspace", className="metric-label")]
    
    return status, blade_freq, confidence, strength, drone_type

@app.callback(
    Output("data-source-radio", "value"),
    [Input("data-source-radio", "value")]
)
def update_data_source(data_source):
    """Update global data source setting when user changes the radio button."""
    global CURRENT_DATA_SOURCE
    CURRENT_DATA_SOURCE = data_source
    
    if data_source == 'synthetic':
        logger.info("[TEST] Synthetic test data enabled")
    else:
        logger.info("[RADAR] Real data from Pi enabled") 
        
    return data_source


@app.callback(
    Output("connection-status", "children"),
    [Input("fast-update-interval", "n_intervals")]
)
def update_connection_status(n):
    """Update connection status indicator."""
    current_time = time.time()
    
    with RADAR_DATA_LOCK:
        last_heartbeat = RADAR_DATA.get('last_heartbeat_time', 0)
        total_points = sum(len(pts) for pts in RADAR_DATA['persistent_points'].values())
        data_source = CURRENT_DATA_SOURCE
    
    # Check if we have recent data
    time_since_data = current_time - last_heartbeat
    
    if data_source == 'synthetic':
        status_text = "[TEST] Test Data Mode"
        status_color = "cyan"
    elif time_since_data < 5.0 and total_points > 0:
        status_text = f"🟢 Pi Connected ({total_points} pts)"
        status_color = "lime"
    elif time_since_data < 10.0:
        status_text = "🟡 Pi Connection Weak"
        status_color = "yellow"
    else:
        status_text = "🔴 Pi Disconnected"
        status_color = "red"
    
    return html.Span(status_text, style={'color': status_color, 'font-size': '14px'})


# ==== SECTION: Application Startup & Cleanup ====
def cleanup_handler(signum=None, frame=None):
    """Cleanup resources on exit."""
    logger.info("Cleaning up resources...")
    logger.info("Cleanup completed.")

if __name__ == "__main__":
    logger.info("--- Starting 4D Radar Dashboard Application ---")

    # Start the UDP packet receiver thread
    receiver_thread_obj = threading.Thread(target=receiver_thread, daemon=True)
    receiver_thread_obj.start()

    # NEW: Start the MAVLink obstacle exporter thread
    mavlink_thread_obj = threading.Thread(target=mavlink_exporter_thread, daemon=True)
    mavlink_thread_obj.start()

    logger.info("Starting Dash server... You should now see the link below!")
    
    app.run(host="0.0.0.0", port=8050, debug=False)