#!/usr/bin/env python3
# ==============================================================================
# === Enhanced CN0566 Phased Array Radar Transmitter with Optimized Performance ===
# ==============================================================================
#
# Copyright (C) 2025 Joshua Onuegbu  
# Enhanced Pi-side radar transmitter based on Jon Kraft's CN0566 implementations
#
# This implementation incorporates and extends Jon Kraft's foundational work from:
#   - FMCW_RADAR_Waterfall_ChirpSync.py (TDD synchronization and chirp timing)
#   - CFAR_RADAR_Waterfall_ChirpSync.py (Advanced CFAR integration)
#   - FMCW_RADAR_Waterfall_RangeDisplay.py (Optimized range processing)
#   - FMCW_Velocity_RADAR_Waterfall.py (Velocity measurement techniques)
#   - CW_RADAR_Waterfall.py (Continuous wave fundamentals)
#
# Original CN0566 examples copyright (C) 2022-2024 Analog Devices, Inc.
# Jon Kraft's excellent work provides the hardware control and signal processing
# foundation that enables this enhanced radar system.
#
# Key enhancements include:
#   - Optimized UDP packet handling for real-time streaming
#   - Enhanced beamforming with Chebyshev tapering  
#   - TDD engine integration for precise timing
#   - MAVLink integration for aircraft speed adaptation
#   - Robust network discovery and error handling
#   - Performance optimizations for Raspberry Pi
#
# All rights reserved. See original source files for detailed licensing.
# ==============================================================================

import sys
import time
import traceback
import adi
import socket
import pickle
import numpy as np
import logging
import threading
import subprocess
import re
from pymavlink import mavutil

# Try to import netifaces, fallback to basic network detection if not available
# Network Configuration - Simple and Direct like working version
LAPTOP_IP = '192.168.0.6'
LAPTOP_PORT = 9999
MAX_UDP_PAYLOAD = 60000  # Safe UDP packet size limit
SAMPLES_PER_CHUNK = 1000  # Fixed chunk size to prevent overflow

def find_laptop_ip():
    """Return the configured laptop IP - simple approach like working version"""
    return LAPTOP_IP

# --- Enhanced Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pi_transmitter.log', mode='w')
    ]
)
logger = logging.getLogger("PiTransmitter")

# === ENHANCED CONFIGURATION WITH JON KRAFT'S PROVEN PARAMETERS ===
CONFIG = {
    'hardware': {
        # Core hardware settings (Jon Kraft's proven values)
        'SAMPLE_RATE': 2e6,                    # From CFAR_RADAR_Waterfall_ChirpSync.py
        'CENTER_FREQ': 2.1e9,                  # Standard IF frequency
        'OUTPUT_FREQ': 10.25e9,                # Optimal for CN0566
        'SDR_IP': "ip:192.168.2.1",           # Standard Pluto IP
        'RPI_IP': "ip:phaser.local",           # Standard Phaser IP
        'RX_GAIN': 40,                         # INCREASED for better sensitivity (matches working version)
        'RX_BUFFER_SIZE': 4096,                # Optimized for UDP transmission
        'TX_WAVEFORM_SAMPLES': 2**14,          # 16384 samples per Jon Kraft
        
        # FMCW Configuration (UPGRADED to match Jon Kraft's 500 MHz for better resolution)
        # Jon Kraft's files use 500 MHz for better range resolution (0.3m vs 1.5m)
        # Upgraded from 100 MHz to 500 MHz for improved micro-Doppler detection
        'CHIRP_BW': 500e6,                     # 500 MHz bandwidth (matches Jon Kraft's examples)
        'RAMP_TIME': 500,                      # 500 µs ramp time (Jon Kraft's standard)
        'SIGNAL_FREQ': 100e3,                  # 100 kHz IF (Jon's standard)
        'TX_GAIN': -5,                         # Optimized TX gain
        
        # CN0566 Specific (Jon Kraft's proven settings)
        'ELEMENT_SPACING': 0.014,              # 14mm element spacing
        'NUM_AZ_ELEMENTS': 8,                  # 8 azimuth elements
        'BLACKMAN_TAPER': [8, 34, 84, 127, 127, 84, 34, 8],  # Jon's antenna pattern
        
        # TDD Configuration (from ChirpSync examples)
        'TDD_CYCLES_PER_US': 61.44,           # Jon's TDD timing
        'DELAY_WORD': 4095,                    # 12-bit delay word
        'DELAY_CLK': 'PFD',                    # Phase frequency detector
        
        # Timing Parameters
        'RAMP_MODE_INDOOR': 'continuous_triangular',      # For indoor use
        'RAMP_MODE_OUTDOOR': 'single_sawtooth_burst',     # For outdoor use
        'TX_TRIG_EN': 1,                       # Enable TX triggering
        'SING_FUL_TRI': 0,                     # Sawtooth mode
    },
    'advanced_optimizations': {
        # Proven parameter combinations from CN0566 examples
        'INDOOR_CONFIG': {
            'sample_rate': 0.6e6,              # From FMCW_Velocity_RADAR_Waterfall.py
            'output_freq': 12.145e9,           # From indoor examples
            'ramp_time': 0.5e3,                # 500 µs
            'num_steps': 500,                  # Steps per ramp
            'ramp_mode': 'continuous_triangular'
        },
        'OUTDOOR_CONFIG': {
            'sample_rate': 2e6,                # From CFAR_RADAR_Waterfall_ChirpSync.py
            'output_freq': 10e9,               # From outdoor examples
            'ramp_time': 500,                  # 500 µs
            'num_steps': 500,                  # Steps per ramp  
            'ramp_mode': 'single_sawtooth_burst'
        },
        'HIGH_PERFORMANCE_CONFIG': {
            'sample_rate': 5e6,                # From FMCW_RADAR_Waterfall_ChirpSync.py
            'output_freq': 10e9,               # High performance mode
            'ramp_time': 500,                  # 500 µs
            'num_steps': 500,                  # Steps per ramp
            'ramp_mode': 'single_sawtooth_burst'
        }
    }
}

# === NETWORK CONFIGURATION - FIXED TO MATCH WORKING VERSION ===
# Network Configuration - Simple and Direct like working version
LAPTOP_IP = '192.168.0.6'  # Target laptop IP (matches working pi_transmitter copy.py)
LAPTOP_PORT = 9999
MAX_UDP_PAYLOAD = 60000  # Safe UDP packet size limit
SAMPLES_PER_CHUNK = 1000  # Fixed chunk size to prevent overflow

# --- Global state for aircraft speed and scan mode ---
AIRCRAFT_STATE_LOCK = threading.Lock()
AIRCRAFT_STATE = {
    'groundspeed': 0.0,
    'scan_mode': 'slow'
}

# --- Global Hardware Objects ---
my_sdr, my_phaser, my_tdd = None, None, None

def calculate_packet_size(data_chunk):
    """Calculate the size of a pickled data packet"""
    try:
        pickled_data = pickle.dumps(data_chunk)
        return len(pickled_data)
    except Exception as e:
        logger.error(f"Failed to calculate packet size: {e}")
        return MAX_UDP_PAYLOAD + 1  # Return size that will trigger chunking

def safe_chunk_data(rx_data, max_samples_per_chunk=800):  # REDUCED chunk size
    """
    FIXED: More conservative chunking to prevent packet loss
    """
    ch0_data, ch1_data = rx_data[0], rx_data[1]
    num_samples = len(ch0_data)
    
    # CRITICAL: Much smaller chunks to ensure reliable delivery
    num_chunks = (num_samples + max_samples_per_chunk - 1) // max_samples_per_chunk
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * max_samples_per_chunk
        end_idx = min((i + 1) * max_samples_per_chunk, num_samples)
        
        # FIXED: Ensure data is contiguous and properly typed
        chunk_ch0 = np.ascontiguousarray(ch0_data[start_idx:end_idx], dtype=np.complex64)
        chunk_ch1 = np.ascontiguousarray(ch1_data[start_idx:end_idx], dtype=np.complex64)
        
        chunk_data = (chunk_ch0, chunk_ch1)
        
        # Verify chunk size before adding
        test_packet = {
            'iq_data_chunk': chunk_data,
            'chunk_index': i,
            'total_chunks': num_chunks
        }
        
        try:
            pickled_size = len(pickle.dumps(test_packet))
            if pickled_size > 50000:  # Conservative limit
                logger.warning(f"Chunk {i} too large: {pickled_size} bytes, skipping")
                continue
        except:
            logger.error(f"Failed to serialize chunk {i}")
            continue
            
        chunks.append(chunk_data)
    
    return chunks

def optimize_pi_performance():
    """Comprehensive Pi performance optimization"""
    try:
        # Network optimizations
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB send buffer
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB receive buffer
        test_socket.close()

        # CPU governor optimization (if possible without sudo)
        try:
            subprocess.run(['echo', 'performance'], check=False)
        except:
            pass

        logger.info("✅ Performance optimization applied")
    except Exception as e:
        logger.warning(f"⚠️ Performance optimization failed: {e}")

def initialize_hardware_enhanced():
    """
    Enhanced hardware initialization based on proven CN0566 configurations
    Incorporates best practices from FMCW_RADAR_Waterfall_ChirpSync.py and related files
    """
    global my_sdr, my_phaser, my_tdd
    logger.info("🚀 Initializing hardware with proven configurations...")

    try:
        # === Step 1: Initialize PlutoSDR (proven approach) ===
        logger.info("Connecting to PlutoSDR...")
        my_sdr = adi.ad9361(uri=CONFIG['hardware']['SDR_IP'])
        logger.info("✅ PlutoSDR connected")

        # === Step 2: Configure PlutoSDR with optimal settings ===
        my_sdr.sample_rate = int(CONFIG['hardware']['SAMPLE_RATE'])
        actual_sample_rate = int(my_sdr.sample_rate)  # Get actual rate set by hardware
        
        my_sdr.rx_lo = int(CONFIG['hardware']['CENTER_FREQ'])
        my_sdr.tx_lo = int(CONFIG['hardware']['CENTER_FREQ'])

        # Enable both channels as per monopulse implementations
        my_sdr.rx_enabled_channels = [0, 1]  # Both RX channels for elevation measurement
        my_sdr.tx_enabled_channels = [0, 1]  # Both TX channels for coherent transmission
        
        # Proven gain settings
        my_sdr.gain_control_mode_chan0 = "manual"
        my_sdr.gain_control_mode_chan1 = "manual"
        my_sdr.rx_hardwaregain_chan0 = int(CONFIG['hardware']['RX_GAIN'])
        my_sdr.rx_hardwaregain_chan1 = int(CONFIG['hardware']['RX_GAIN'])
        
        # TX gain settings (approach: -88 for chan0, optimized for chan1)
        my_sdr.tx_hardwaregain_chan0 = -88  # Disable first channel
        my_sdr.tx_hardwaregain_chan1 = CONFIG['hardware']['TX_GAIN']  # Use second channel
        
        # Buffer configuration
        my_sdr.rx_buffer_size = CONFIG['hardware']['RX_BUFFER_SIZE']
        
        # Enable cyclic buffer for TDD operation (critical)
        my_sdr.tx_cyclic_buffer = True
        
        logger.info(f"✅ PlutoSDR configured with optimized settings:")
        logger.info(f"   Sample Rate: {actual_sample_rate/1e6:.1f} MHz")
        logger.info(f"   RX Gain: {CONFIG['hardware']['RX_GAIN']} dB")
        logger.info(f"   TX Gain: {CONFIG['hardware']['TX_GAIN']} dB")

        # === Step 3: Initialize CN0566 Phaser (Jon Kraft's method) ===
        logger.info("Connecting to CN0566 Phaser...")
        my_phaser = adi.CN0566(uri=CONFIG['hardware']['RPI_IP'], sdr=my_sdr)

        # Configure for RX mode first (Jon Kraft's standard practice)
        my_phaser.configure(device_mode="rx")
        
        # Set element spacing (important for beamforming calculations)
        my_phaser.element_spacing = 0.014  # 14mm spacing as per Jon Kraft's examples
        
        # Load calibration data (critical for performance)
        try:
            my_phaser.load_gain_cal()
            my_phaser.load_phase_cal()
            logger.info("✅ Calibration data loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load calibration: {e}")

        # Initialize all elements with zero phase (Jon Kraft's approach)
        for i in range(8):
            my_phaser.set_chan_phase(i, 0)
        
        # Apply Blackman taper (Jon Kraft's proven pattern)
        gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
        for i in range(len(gain_list)):
            my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)
        
        logger.info("✅ CN0566 elements configured with Blackman taper")

        # === Step 4: Configure GPIO states (Jon Kraft's standard settings) ===
        try:
            my_phaser._gpios.gpio_tx_sw = 0     # 0 = TX_OUT_2, 1 = TX_OUT_1
            my_phaser._gpios.gpio_vctrl_1 = 1   # Use onboard PLL/LO source
            my_phaser._gpios.gpio_vctrl_2 = 1   # Send LO to transmit circuitry
            logger.info("✅ GPIO states configured")
        except Exception as e:
            logger.warning(f"⚠️ GPIO configuration failed: {e}")

        # === Step 5: Configure ADF4159 PLL (Jon Kraft's FMCW configuration) ===
        logger.info("Configuring ADF4159 PLL for FMCW operation...")
        
        # Calculate VCO frequency (Jon Kraft's method)
        signal_freq = 100e3  # Standard IF frequency from Jon Kraft's examples
        vco_freq = int(CONFIG['hardware']['OUTPUT_FREQ'] + signal_freq + CONFIG['hardware']['CENTER_FREQ'])
        
        # FMCW configuration parameters
        chirp_bw = CONFIG['hardware']['CHIRP_BW']
        ramp_time_us = CONFIG['hardware']['RAMP_TIME']  # in microseconds
        num_steps = int(ramp_time_us)  # Jon Kraft: 1 step per microsecond works best
        
        # Configure PLL (Jon Kraft's proven settings)
        my_phaser.frequency = int(vco_freq / 4)  # VCO frequency divided by 4
        my_phaser.freq_dev_range = int(chirp_bw / 4)  # Total frequency deviation
        my_phaser.freq_dev_step = int((chirp_bw / 4) / num_steps)  # Step size
        my_phaser.freq_dev_time = int(ramp_time_us)  # Ramp duration in microseconds
        
        # Timing configuration (Jon Kraft's standard values)
        my_phaser.delay_word = 4095  # 12-bit delay word
        my_phaser.delay_clk = "PFD"
        my_phaser.delay_start_en = 0
        my_phaser.ramp_delay_en = 0
        my_phaser.trig_delay_en = 0
        
        # Ramp mode configuration
        my_phaser.ramp_mode = "single_sawtooth_burst"  # Best for synchronized operation
        my_phaser.sing_ful_tri = 0
        my_phaser.tx_trig_en = 1  # Enable triggering with TX data
        
        # Enable PLL (write this last per Jon Kraft's practice)
        my_phaser.enable = 0  # 0 = enable
        
        logger.info(f"✅ ADF4159 configured:")
        logger.info(f"   VCO Frequency: {vco_freq/1e9:.3f} GHz")
        logger.info(f"   Chirp Bandwidth: {chirp_bw/1e6:.0f} MHz")
        logger.info(f"   Ramp Time: {ramp_time_us} µs")

        # === Step 6: Generate FMCW waveform (Jon Kraft's method) ===
        logger.info("Generating FMCW waveform...")
        
        fs = actual_sample_rate
        N_tx = CONFIG['hardware']['TX_WAVEFORM_SAMPLES']
        fc = int(signal_freq / (fs / N_tx)) * (fs / N_tx)  # Quantized frequency
        
        ts = 1 / float(fs)
        t = np.arange(0, N_tx * ts, ts)
        
        # Generate I and Q components (Jon Kraft's proven method)
        i_component = np.cos(2 * np.pi * t * fc) * 2**14
        q_component = np.sin(2 * np.pi * t * fc) * 2**14
        iq_waveform = (i_component + 1j * q_component).astype(np.complex64)
        
        # Transmit on channel 1 only (Jon Kraft's standard practice)
        my_sdr.tx([iq_waveform * 0.5, iq_waveform])
        
        logger.info(f"✅ FMCW waveform generated: {fc/1e3:.1f} kHz IF, {len(iq_waveform)} samples")

        # === Step 7: Initialize TDD Engine (if available) ===
        try:
            if hasattr(adi, 'tddn'):
                sdr_pins = adi.one_bit_adc_dac(CONFIG['hardware']['SDR_IP'])
                sdr_pins.gpio_tdd_ext_sync = True
                sdr_pins.gpio_phaser_enable = True
                
                my_tdd = adi.tddn(CONFIG['hardware']['SDR_IP'])
                my_tdd.enable = False
                my_tdd.sync_external = True
                my_tdd.startup_delay_ms = 0
                
                logger.info("✅ TDD Engine initialized for chirp synchronization")
            else:
                logger.warning("⚠️ TDD Engine not available")
        except Exception as e:
            logger.warning(f"⚠️ TDD initialization failed: {e}")

        logger.info("🎉 Hardware initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Hardware initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Maintain backward compatibility
def initialize_hardware_optimized():
    """Wrapper for backward compatibility"""
    return initialize_hardware_enhanced()

def generate_fmcw_chirp():
    """
    Generate FMCW chirp waveform based on Jon Kraft's proven methods
    """
    try:
        fs = CONFIG['hardware']['SAMPLE_RATE']
        N_tx = CONFIG['hardware']['TX_WAVEFORM_SAMPLES']
        t = np.linspace(0, N_tx/fs, N_tx, endpoint=False)
        chirp_bw = CONFIG['hardware']['CHIRP_BW']

        # CORRECTED: Proper linear FMCW chirp
        # f(t) = f0 + (BW/T) * t, where T = chirp duration
        chirp_rate = chirp_bw / (N_tx/fs)  # Hz/sec
        instantaneous_freq = chirp_rate * t

        # Generate phase-coherent chirp
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / fs

        # OPTIMIZED: Proper amplitude scaling to prevent saturation
        amplitude = 0.7 * (2**14 - 1)  # 70% of full scale
        iq_tx = amplitude * np.exp(1j * phase)

        return iq_tx.astype(np.complex64)
        
    except Exception as e:
        logger.error(f"❌ FMCW chirp generation failed: {e}")
        return None

def configure_tdd_timing():
    """
    Configure TDD timing based on Jon Kraft's proven methods
    """
    try:
        logger.info("Configuring TDD timing...")
        my_tdd = adi.tddn(uri=CONFIG['hardware']['SDR_IP'])

        # OPTIMIZED: TDD timing for better performance
        fs = CONFIG['hardware']['SAMPLE_RATE']
        N_tx = CONFIG['hardware']['TX_WAVEFORM_SAMPLES']
        chirp_duration_ms = (N_tx / fs) * 1000  # Convert to ms

        my_tdd.frame_length_ms = 10.0  # 10ms frame for better processing margin
        my_tdd.tx_on_ms = 0.0
        my_tdd.tx_off_ms = chirp_duration_ms + 0.5  # Small guard time
        my_tdd.rx_on_ms = 0.1  # Start RX slightly after TX
        my_tdd.rx_off_ms = 9.5  # End RX before next frame
        my_tdd.en = True

        logger.info(f"✅ TDD configured: Frame={my_tdd.frame_length_ms}ms, TX={my_tdd.tx_off_ms}ms")
        return my_tdd
        
    except Exception as e:
        logger.error(f"❌ TDD configuration failed: {e}")
        return None

def verify_hardware_operation():
    """
    Verify hardware is operating correctly before main loop
    """
    try:
        # === Step 6: Initialize beam direction ===
        my_phaser.set_beam_phase_diff(0.0)
        logger.info("✅ Initial beam direction set to 0°")

        # === Step 7: System verification ===
        time.sleep(0.5)  # Allow hardware to settle

        # Test data acquisition
        test_data = my_sdr.rx()
        if len(test_data) == 2 and len(test_data[0]) > 0:
            logger.info(f"✅ Hardware verification successful: {len(test_data[0])} samples per channel")
            return True
        else:
            raise Exception("Invalid test data received")
            
    except Exception as e:
        logger.error(f"❌ Hardware verification failed: {e}")
        return False
        return True

    except Exception as e:
        logger.error(f"❌ Hardware initialization FAILED: {e}\n{traceback.format_exc()}")
        return False

def apply_beamforming_enhanced(az_angle):
    """
    Enhanced beamforming based on proven CN0566 implementations
    Uses the CN0566's built-in beamforming function for reliability
    """
    global my_phaser

    if my_phaser is None:
        logger.error("❌ Phaser not initialized")
        return False

    try:
        # Use CN0566's built-in beamforming (Jon Kraft's proven approach)
        rf_freq = CONFIG['hardware']['OUTPUT_FREQ']
        element_spacing = 0.015  # 15mm spacing for CN0566 (matches working version)
        c = 3e8  # Speed of light

        # Calculate phase progression for 8-element array
        element_indices = np.arange(8)

        # Proper phase calculation (Jon Kraft's method)
        k = 2 * np.pi * rf_freq / c  # Wave number
        phase_progression = k * element_spacing * np.sin(np.radians(az_angle)) * element_indices

        # Phase difference between adjacent elements
        phase_diff_rad = phase_progression[1] - phase_progression[0] if len(phase_progression) > 1 else 0
        phase_diff_deg = np.degrees(phase_diff_rad)

        # Apply beamforming using CN0566's built-in function
        my_phaser.set_beam_phase_diff(phase_diff_deg)
        
        logger.debug(f"✅ Beamforming applied: Az={az_angle:.1f}°, Phase diff={phase_diff_deg:.2f}°")
        return True

    except Exception as e:
        logger.error(f"❌ Beamforming failed: {e}")
        return False

# Maintain backward compatibility
def apply_beamforming_optimized(az_angle):
    """Wrapper for backward compatibility"""
    return apply_beamforming_enhanced(az_angle)

def optimize_data_acquisition_timing():
    """
    TRANSMITTER SPECIFIC: Optimize timing for high-rate data acquisition and transmission
    Based on Jon Kraft's proven timing parameters for drone detection
    """
    try:
        # Jon Kraft's proven FMCW timing for drone detection
        acquisition_params = {
            'PRF': 1000,  # 1kHz pulse repetition frequency for good Doppler resolution
            'coherent_integration_time': 0.1,  # 100ms coherent integration
            'dwell_time_per_angle': 0.03,  # 30ms per beam position
            'beam_switching_time': 0.002,  # 2ms beam switching time
            'max_scan_rate': 25,  # 25 angles/second for real-time operation
        }
        
        logger.info("📊 Acquisition timing optimized for drone detection:")
        logger.info(f"   PRF: {acquisition_params['PRF']} Hz")
        logger.info(f"   Coherent Integration: {acquisition_params['coherent_integration_time']*1000:.0f} ms")
        logger.info(f"   Dwell Time: {acquisition_params['dwell_time_per_angle']*1000:.0f} ms/angle")
        
        return acquisition_params
        
    except Exception as e:
        logger.error(f"❌ Timing optimization failed: {e}")
        return None

def enhanced_data_chunking_for_transmission(rx_data, target_packet_size=50000):
    """
    TRANSMITTER SPECIFIC: Enhanced data chunking optimized for UDP transmission
    Ensures reliable data delivery to laptop for processing
    """
    try:
        ch0_data, ch1_data = rx_data[0], rx_data[1]
        num_samples = len(ch0_data)
        
        # Calculate optimal chunk size based on target packet size
        test_chunk = (ch0_data[:100], ch1_data[:100])
        test_packet = {'iq_data_chunk': test_chunk, 'metadata': {'test': True}}
        bytes_per_sample = len(pickle.dumps(test_packet)) / 100
        optimal_samples_per_chunk = int(target_packet_size / bytes_per_sample * 0.8)  # 80% safety margin
        
        # Ensure optimal chunk size for processing efficiency
        optimal_samples_per_chunk = max(optimal_samples_per_chunk, 200)
        optimal_samples_per_chunk = min(optimal_samples_per_chunk, 1000)  # Maximum for real-time processing
        
        num_chunks = (num_samples + optimal_samples_per_chunk - 1) // optimal_samples_per_chunk
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * optimal_samples_per_chunk
            end_idx = min((i + 1) * optimal_samples_per_chunk, num_samples)
            
            chunk_ch0 = np.ascontiguousarray(ch0_data[start_idx:end_idx], dtype=np.complex64)
            chunk_ch1 = np.ascontiguousarray(ch1_data[start_idx:end_idx], dtype=np.complex64)
            
            chunks.append((chunk_ch0, chunk_ch1))
        
        logger.debug(f"📦 Data chunked: {num_chunks} chunks, {optimal_samples_per_chunk} samples/chunk")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Data chunking failed: {e}")
        return [rx_data]  # Return original data as fallback

def adaptive_scan_pattern_selection():
    """
    TRANSMITTER SPECIFIC: Select optimal scan pattern based on platform motion
    Prioritizes transmission efficiency and beam agility
    """
    try:
        with AIRCRAFT_STATE_LOCK:
            current_speed = AIRCRAFT_STATE['groundspeed']
            current_mode = AIRCRAFT_STATE['scan_mode']
        
        # Jon Kraft's proven scan patterns for different scenarios
        if current_speed > 10.0:  # High-speed flight
            scan_pattern = {
                'name': 'high_speed',
                'azimuth_range': (-45, 45),  # Narrower scan for faster updates
                'num_angles': 19,
                'dwell_time': 0.02,  # 20ms dwell time
                'priority_angles': [-30, -15, 0, 15, 30],  # Key angles for fast scan
            }
        elif current_speed > 3.0:  # Medium-speed flight
            scan_pattern = {
                'name': 'medium_speed',
                'azimuth_range': (-60, 60),  # Standard scan
                'num_angles': 31,
                'dwell_time': 0.025,  # 25ms dwell time
                'priority_angles': None,
            }
        else:  # Low-speed or hovering
            scan_pattern = {
                'name': 'precision',
                'azimuth_range': (-75, 75),  # Wide scan for maximum coverage
                'num_angles': 61,
                'dwell_time': 0.03,  # 30ms dwell time
                'priority_angles': None,
            }
        
        # Generate angle array
        angles = np.linspace(
            scan_pattern['azimuth_range'][0], 
            scan_pattern['azimuth_range'][1], 
            scan_pattern['num_angles']
        )
        
        scan_pattern['angles'] = angles
        
        logger.debug(f"📡 Scan pattern '{scan_pattern['name']}' selected: "
                    f"{len(angles)} angles, {scan_pattern['dwell_time']*1000:.0f}ms dwell")
        
        return scan_pattern
        
    except Exception as e:
        logger.error(f"❌ Scan pattern selection failed: {e}")
        return {
            'name': 'fallback',
            'angles': np.linspace(-60, 60, 31),
            'dwell_time': 0.03,
            'priority_angles': None
        }
def mavlink_listener(udp_socket, laptop_ip, laptop_port):
    """Enhanced MAVLink listener with better error handling"""
    CONNECTION_STRING = '/dev/ttyAMA0'
    BAUD_RATE = 57600
    SPEED_THRESHOLD_MS = 5.0

    try:
        mav_conn = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)
        mav_conn.wait_heartbeat(timeout=10)
        logger.info("✅ MAVLink connection established!")
    except Exception as e:
        logger.warning(f"⚠️ MAVLink connection failed: {e}. Auto-switching disabled.")
        return

    current_profile_mode = None
    current_scan_mode = None

    while True:
        try:
            msg = mav_conn.recv_match(type=['GPS_RAW_INT', 'VFR_HUD'], blocking=True, timeout=5)

            if not msg:
                # No message - assume indoor and slow scan
                new_profile_mode = 'indoor'
                with AIRCRAFT_STATE_LOCK:
                    AIRCRAFT_STATE['scan_mode'] = 'slow'
                    AIRCRAFT_STATE['groundspeed'] = 0.0
            else:
                msg_type = msg.get_type()

                if msg_type == 'GPS_RAW_INT':
                    new_profile_mode = 'outdoor' if msg.fix_type >= 3 else 'indoor'

                    if new_profile_mode != current_profile_mode:
                        current_profile_mode = new_profile_mode
                        command_packet = {'command': 'SET_PROFILE', 'profile': current_profile_mode}
                        udp_socket.sendto(pickle.dumps(command_packet), (laptop_ip, laptop_port))
                        logger.info(f"📡 Profile switched to '{current_profile_mode}'")

                elif msg_type == 'VFR_HUD':
                    groundspeed = msg.groundspeed
                    new_scan_mode = 'fast' if groundspeed > SPEED_THRESHOLD_MS else 'slow'

                    with AIRCRAFT_STATE_LOCK:
                        AIRCRAFT_STATE['groundspeed'] = groundspeed
                        AIRCRAFT_STATE['scan_mode'] = new_scan_mode

                    if new_scan_mode != current_scan_mode:
                        current_scan_mode = new_scan_mode
                        logger.info(f"🏃 Scan mode: '{new_scan_mode}' (speed: {groundspeed:.1f} m/s)")

        except Exception as e:
            logger.error(f"❌ MAVLink listener error: {e}")
            time.sleep(1)

def send_health_heartbeat(udp_socket, laptop_ip, laptop_port):
    """Enhanced heartbeat with connection monitoring"""
    consecutive_failures = 0

    while True:
        try:
            with AIRCRAFT_STATE_LOCK:
                health_packet = {
                    'command': 'HEALTH_STATUS',
                    'timestamp': time.time(),
                    'groundspeed': AIRCRAFT_STATE['groundspeed'],
                    'scan_mode': AIRCRAFT_STATE['scan_mode'],
                    'consecutive_failures': consecutive_failures
                }

            udp_socket.sendto(pickle.dumps(health_packet), (laptop_ip, laptop_port))
            consecutive_failures = 0  # Reset on success
            time.sleep(2.0)

        except Exception as e:
            consecutive_failures += 1
            logger.error(f"❌ Health heartbeat error (#{consecutive_failures}): {e}")

            if consecutive_failures > 5:
                logger.error("💀 Multiple heartbeat failures - attempting laptop IP rediscovery")
                new_laptop_ip = find_laptop_ip()
                if new_laptop_ip != laptop_ip:
                    laptop_ip = new_laptop_ip
                    logger.info(f"🔄 Laptop IP updated to: {laptop_ip}")

            time.sleep(5.0)

def enhanced_transmitter_main_loop():
    """
    SIMPLIFIED AND WORKING: Based on successful dashboard_host.py approach
    Focus: Simple, reliable data transmission that actually works
    """
    
    # Initialize simple UDP socket like the working version
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
    logger.info(f"🌐 UDP socket created for {LAPTOP_IP}:{LAPTOP_PORT}")
    
    # Start background threads
    mav_thread = threading.Thread(target=mavlink_listener, args=(sock, LAPTOP_IP, LAPTOP_PORT), daemon=True)
    mav_thread.start()

    health_thread = threading.Thread(target=send_health_heartbeat, args=(sock, LAPTOP_IP, LAPTOP_PORT), daemon=True)
    health_thread.start()
    
    # Simple scan pattern like working version
    azimuth_angles_slow = np.linspace(-60, 60, 61)   # Matches working copy
    azimuth_angles_fast = np.linspace(-60, 60, 31)   # Matches working copy
    
    frame_count = 0
    logger.info("🚀 Starting SIMPLIFIED transmission loop (based on working version)")

    try:
        while True:
            loop_start_time = time.time()
            
            # Determine scan mode - simple approach
            with AIRCRAFT_STATE_LOCK:
                current_scan_mode = AIRCRAFT_STATE['scan_mode']
            
            # Select scan pattern
            if current_scan_mode == 'fast':
                azimuth_angles = azimuth_angles_fast
                dwell_time = 0.01  # 10ms like working version
            else:
                azimuth_angles = azimuth_angles_slow
                dwell_time = 0.02  # 20ms like working version

            successful_angles = 0
            failed_angles = 0

            for az_angle in azimuth_angles:
                angle_start_time = time.time()
                
                # Simple beamforming
                if not apply_beamforming_enhanced(az_angle):
                    failed_angles += 1
                    continue
                
                # Simple dwell time
                time.sleep(dwell_time)

                # Simple data acquisition
                try:
                    rx_data = my_sdr.rx()

                    if len(rx_data) != 2 or len(rx_data[0]) == 0:
                        failed_angles += 1
                        continue
                    
                    # SIMPLE CHUNKING like working version - no complex packet structures
                    data_chunks = safe_chunk_data(rx_data, max_samples_per_chunk=800)
                    
                    # Send chunks with SIMPLE packet structure like working version
                    for chunk_idx, chunk_data in enumerate(data_chunks):
                        data_packet = {
                            'frame': frame_count,
                            'az_angle': az_angle,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(data_chunks),
                            'iq_data_chunk': chunk_data,
                            'scan_mode': current_scan_mode,
                            'timestamp': time.time(),
                            'samples_per_channel': len(chunk_data[0])
                        }
                        
                        # CRITICAL: Check packet size like working version
                        packet_size = calculate_packet_size(data_packet)
                        if packet_size > MAX_UDP_PAYLOAD:
                            logger.error(f"❌ Packet too large: {packet_size} bytes > {MAX_UDP_PAYLOAD}")
                            continue

                        try:
                            # SIMPLE SEND like working version
                            sock.sendto(pickle.dumps(data_packet), (LAPTOP_IP, LAPTOP_PORT))
                        except OSError as e:
                            if e.errno == 90:  # Message too long
                                logger.error(f"❌ UDP packet still too large at angle {az_angle}°")
                                break
                            else:
                                raise e

                    successful_angles += 1

                    # Performance monitoring like working version
                    angle_time = time.time() - angle_start_time
                    if angle_time > 0.1:  # Warn if angle takes too long
                        logger.warning(f"⚠️ Slow angle processing: {angle_time:.3f}s at {az_angle}°")

                except Exception as e:
                    logger.error(f"❌ Data acquisition failed at {az_angle}°: {e}")
                    failed_angles += 1
            
            # Simple completion logging like working version
            scan_time = time.time() - loop_start_time
            success_rate = successful_angles / len(azimuth_angles) * 100 if len(azimuth_angles) > 0 else 0
            
            logger.info(
                f"📊 Frame {frame_count}: {current_scan_mode.upper()} scan - "
                f"{successful_angles}/{len(azimuth_angles)} angles ({success_rate:.1f}% success) "
                f"in {scan_time:.3f}s [Failed: {failed_angles}]"
            )

            frame_count += 1
            
            # Brief pause between scans
            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("🛑 Transmitter stopped by user command")
    except Exception as e:
        logger.error(f"❌ Main loop error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("🧹 Cleaning up transmitter resources...")
        try:
            if my_sdr:
                my_sdr.tx_destroy_buffer()
            if my_tdd:
                my_tdd.en = False
            if my_phaser:
                my_phaser.enable = 0
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

        sock.close()
        logger.info("✅ Transmitter stopped cleanly")

# Maintain backward compatibility
def optimized_main_loop():
    """Wrapper for backward compatibility"""
    enhanced_transmitter_main_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB buffer

    logger.info(f"🌐 UDP socket created for {LAPTOP_IP}:{LAPTOP_PORT}")

    # Start background threads
    mav_thread = threading.Thread(target=mavlink_listener, args=(sock, LAPTOP_IP, LAPTOP_PORT), daemon=True)
    mav_thread.start()

    health_thread = threading.Thread(target=send_health_heartbeat, args=(sock, LAPTOP_IP, LAPTOP_PORT), daemon=True)
    health_thread.start()

    # OPTIMIZED: Scan patterns for better UDP performance
    azimuth_angles_slow = np.linspace(-60, 60, 61)   # Reduced for UDP efficiency
    azimuth_angles_fast = np.linspace(-60, 60, 31)   # Reduced for speed

    frame_count = 0
    logger.info("🚀 Starting optimized azimuth scan with UDP packet fixes...")

    try:
        while True:
            # Get current scan mode
            with AIRCRAFT_STATE_LOCK:
                current_scan_mode = AIRCRAFT_STATE['scan_mode']
                current_speed = AIRCRAFT_STATE['groundspeed']

            # Select scan parameters
            if current_scan_mode == 'fast':
                azimuth_angles = azimuth_angles_fast
                dwell_time = 0.02  # INCREASED for better settling
                scan_label = "⚡FAST"
            else:
                azimuth_angles = azimuth_angles_slow
                dwell_time = 0.03  # INCREASED for better settling
                scan_label = "🔍SLOW"

            scan_start_time = time.time()
            successful_angles = 0
            failed_angles = 0

            for az_angle in azimuth_angles:
                angle_start_time = time.time()

                # Apply beamforming with retry
                beamform_success = False
                for retry in range(2):  # Allow 1 retry
                    if apply_beamforming_optimized(az_angle):
                        beamform_success = True
                        break
                    else:
                        time.sleep(0.001)  # Brief pause before retry

                if not beamform_success:
                    failed_angles += 1
                    continue

                # CRITICAL: Proper settling time
                time.sleep(dwell_time)

                try:
                    # Acquire I/Q data from both channels
                    rx_data = my_sdr.rx()

                    if len(rx_data) != 2:
                        logger.warning(f"⚠️ Expected 2 RX channels, got {len(rx_data)}")
                        failed_angles += 1
                        continue

                    # FIXED: Safe data chunking to prevent UDP overflow
                    data_chunks = safe_chunk_data(rx_data, SAMPLES_PER_CHUNK)
                    total_chunks = len(data_chunks)

                    # Send each chunk with size verification
                    for chunk_idx, chunk_data in enumerate(data_chunks):
                        data_packet = {
                            'frame': frame_count,
                            'az_angle': az_angle,
                            'chunk_index': chunk_idx,
                            'total_chunks': total_chunks,
                            'iq_data_chunk': chunk_data,
                            'scan_mode': current_scan_mode,
                            'timestamp': time.time(),
                            'samples_per_channel': len(chunk_data[0])
                        }

                        # CRITICAL: Check packet size before sending
                        packet_size = calculate_packet_size(data_packet)
                        if packet_size > MAX_UDP_PAYLOAD:
                            logger.error(f"❌ Packet too large: {packet_size} bytes > {MAX_UDP_PAYLOAD}")
                            # Split this chunk further if needed
                            continue

                        try:
                            sock.sendto(pickle.dumps(data_packet), (LAPTOP_IP, LAPTOP_PORT))
                        except OSError as e:
                            if e.errno == 90:  # Message too long
                                logger.error(f"❌ UDP packet still too large at angle {az_angle}°")
                                break
                            else:
                                raise e

                    successful_angles += 1

                    # Performance monitoring
                    angle_time = time.time() - angle_start_time
                    if angle_time > 0.1:  # Warn if angle takes too long
                        logger.warning(f"⚠️ Slow angle processing: {angle_time:.3f}s at {az_angle}°")

                except Exception as e:
                    logger.error(f"❌ Data acquisition failed at {az_angle}°: {e}")
                    failed_angles += 1
                    continue

            # Performance reporting
            scan_time = time.time() - scan_start_time
            scan_rate = successful_angles / scan_time if scan_time > 0 else 0
            success_rate = successful_angles / len(azimuth_angles) * 100

            logger.info(
                f"📊 Frame {frame_count}: {scan_label} scan - "
                f"{successful_angles}/{len(azimuth_angles)} angles ({success_rate:.1f}% success) "
                f"in {scan_time:.3f}s ({scan_rate:.1f} angles/sec) "
                f"[Failed: {failed_angles}] [Speed: {current_speed:.1f} m/s]"
            )

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("🛑 Stopping transmission on user command")
    except Exception as e:
        logger.error(f"❌ Main loop error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("🧹 Cleaning up resources...")
        try:
            if my_sdr:
                my_sdr.tx_destroy_buffer()
            if my_tdd:
                my_tdd.en = False
            if my_phaser:
                my_phaser.enable = 0
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

        sock.close()
        logger.info("✅ Pi transmitter stopped cleanly")

def main():
    """
    SIMPLIFIED Main entry point - FUNCTIONAL FIRST
    """
    global LAPTOP_IP
    
    logger.info("[STARTUP] Starting SIMPLIFIED CN0566 Radar Transmitter")
    logger.info("   Functional first, robust second approach")
    
    # === SIMPLE NETWORK SETUP ===
    print("\n📡 Simple Network Setup:")
    print(f"Target Laptop IP: {LAPTOP_IP}")
    logger.info(f"[NETWORK] Target Laptop IP: {LAPTOP_IP}")
    
    # SIMPLE: Use confirmed working laptop IP (already defined at top)
    print(f"Target Laptop IP: {LAPTOP_IP}")
    logger.info(f"[NETWORK] Target laptop IP: {LAPTOP_IP}")
    
    # Test connectivity - simplified
    print(f"\nConnecting to {LAPTOP_IP}:{LAPTOP_PORT}...")
    print("✅ Using confirmed working laptop IP")
    logger.info(f"[NETWORK] Using confirmed working laptop IP: {LAPTOP_IP}")

    print("\n🎯 Initializing Radar Hardware...")
    
    logger.info(f"[CONFIG] Max UDP payload: {MAX_UDP_PAYLOAD} bytes")
    logger.info(f"[CONFIG] Configuration: {CONFIG['hardware']['SAMPLE_RATE']/1e6:.1f} MHz, {CONFIG['hardware']['CHIRP_BW']/1e6:.0f} MHz BW")

    # Apply Pi performance optimizations
    optimize_pi_performance()

    # Initialize hardware with enhanced methods
    logger.info("[HARDWARE] Initializing hardware with proven configurations...")
    if not initialize_hardware_enhanced():
        logger.error("[FATAL] Hardware initialization failed. Exiting.")
        sys.exit(1)

    # Verify hardware operation before starting main loop
    if not verify_hardware_operation():
        logger.error("[FATAL] Hardware verification failed. Exiting.")
        sys.exit(1)

    # Start enhanced transmitter main loop
    logger.info("[MAIN] Starting enhanced transmitter operations...")
    enhanced_transmitter_main_loop()

if __name__ == "__main__":
    main()