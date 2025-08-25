# Development of a 4D Radar Processing System for Small UAV Detection

# Enhanced 4D Radar System for Small UAV Detection

## Table of Contents

*   [Key Features](#key-features)
*   [Authors and Acknowledgements](#authors-and-acknowledgements)
*   [1. Introduction](#1-introduction)
    *   [1.1 Project Motivation](#11-project-motivation)
    *   [1.2 System Overview](#12-system-overview)
    *   [1.3 Software Architecture](#13-software-architecture)
*   [2. Raspberry Pi Data Acquisition Module (`pi_transmitter.py`)](#2-raspberry-pi-data-acquisition-module-pi_transmitterpy)
    *   [2.1 Module Overview](#21-module-overview)
    *   [2.2 Hardware Configuration](#22-hardware-configuration)
    *   [2.3 Beamforming Implementation](#23-beamforming-implementation)
    *   [2.4 Data Transmission Approach](#24-data-transmission-approach)
    *   [2.5 MAVLink Integration (Experimental)](#25-mavlink-integration-experimental)
*   [3. Signal Processing and Visualization Platform (`radar_dashboard.py`)](#3-signal-processing-and-visualization-platform-radar_dashboardpy)
    *   [3.1 Module Overview](#31-module-overview)
    *   [3.2 FMCW Signal Processing Implementation](#32-fmcw-signal-processing-implementation)
    *   [3.3 Target Detection Algorithm Development](#33-target-detection-algorithm-development)
    *   [3.4 Micro-Doppler Analysis Development](#34-micro-doppler-analysis-development)
    *   [3.5 Multi-Target Tracking Development](#35-multi-target-tracking-development)
    *   [3.6 Object Classification Research](#36-object-classification-research)
*   [4. User Interface and Visualization Development](#4-user-interface-and-visualization-development)
    *   [4.1 Dashboard Design Approach](#41-dashboard-design-approach)
    *   [4.2 3D Visualization Implementation](#42-3d-visualization-implementation)
    *   [4.3 Control Interface Development](#43-control-interface-development)
    *   [4.4 Diagnostic Visualization Tools](#44-diagnostic-visualization-tools)

This repository details the development and technical analysis of an advanced 4D radar processing system, specifically designed to enhance the detection capabilities of small Unmanned Aerial Vehicles (UAVs). This project leverages a distributed architecture, featuring a Raspberry Pi for radar control and data acquisition, and a laptop-based platform for sophisticated signal processing and real-time visualization.

Built upon the robust CN0566 reference designs from Analog Devices, this system explores cutting-edge techniques including micro-Doppler analysis, multi-target tracking, and dynamic visualization to address the unique challenges of small UAV detection. It serves as a practical demonstration of utilizing commercial radar development platforms for innovative research and educational purposes.

## Key Features

*   **Distributed Architecture:** Separates data acquisition (Raspberry Pi) from signal processing and visualization (laptop) for optimized performance.
*   **Advanced Signal Processing:** Incorporates real-time FMCW processing, CFAR detection, micro-Doppler analysis, and multi-target tracking algorithms.
*   **Phased Array Beamforming:** Utilizes the CN0566's capabilities for electronic beam steering and azimuth scanning.
*   **Real-time Visualization:** Provides dynamic 3D visualization and a user-friendly interface for radar data.
*   **Experimental MAVLink Integration:** Explores adaptive radar operation based on platform motion and telemetry.
*   **Educational and Research Focus:** Designed as a platform for investigating small UAV detection challenges and radar system development.

## Authors and Acknowledgements

**Developed by:** Joshua Onuegbu (UH Senior Design Team Lead (Spring 2025): Beamforming)
**Team Members:** Mohammad Alkildar, Joseph Lee, Martin Tran, and Richard Truong

**Acknowledgements:**
*   **Jon Kraft, Analog Devices Inc.:** For foundational work on CN0566 algorithms and providing essential Python examples that served as a basis for this enhanced system.
*   **Dr. Aaron Becker, University of Houston:** For his invaluable guidance and support as the faculty sponsor of this project.

---

## 1. Introduction

### 1.1 Project Motivation

The widespread use of small Unmanned Aerial Vehicles (UAVs) in various civilian and commercial sectors has introduced significant challenges for airspace monitoring and security. Unlike larger aircraft, small UAVs often possess a limited radar cross-section and exhibit unique motion characteristics, making them difficult to detect with traditional radar systems. This project aims to address these challenges by investigating the application of commercial radar development platforms for advanced small UAV detection research.

### 1.2 System Overview

The current implementation utilizes the Analog Devices CN0566 phased array radar development platform as the hardware foundation. The system architecture separates data acquisition and signal processing functions across two computing platforms to explore the trade-offs between real-time processing constraints and algorithm complexity.

**Hardware Components:**

- ADALM-PLUTO Software-Defined Radio (SDR)
- CN0566 Phased Array Antenna Board (8-element)
- ADF4159 Phase-Locked Loop for FMCW synthesis
- Raspberry Pi 4 (embedded control platform)
- Laptop computer (signal processing platform)

### 1.3 Software Architecture

The current software implementation employs a distributed processing approach:

```
┌─────────────────┐    UDP Network    ┌──────────────────┐
│  Raspberry Pi   │ ◄──────────────► │   Laptop PC      │
│  (Data Acq.)    │                  │  (Processing)    │
│                 │                  │                  │
│ • Hardware Ctrl │                  │ • Signal Proc.   │
│ • Beamforming   │                  │ • Visualization  │
│ • Data Acq.     │                  │ • Classification │
│ • MAVLink       │                  │ • Tracking       │
└─────────────────┘                  └──────────────────┘
```

This architecture allows for investigation of computationally intensive signal processing algorithms on the laptop platform while maintaining real-time data acquisition on the embedded system.

---

## 2. Raspberry Pi Data Acquisition Module (`pi_transmitter.py`)

### 2.1 Module Overview

The Raspberry Pi component handles hardware control and data acquisition functions. The current implementation focuses on establishing reliable data collection and transmission capabilities while exploring various radar operating modes.

**Primary Functions:**

1. Hardware initialization and configuration
2. FMCW waveform generation and transmission
3. Phased array beam steering control
4. Real-time I/Q data acquisition
5. Network-based data transmission
6. MAVLink protocol integration (experimental)

### 2.2 Hardware Configuration

The current implementation builds upon established CN0566 reference designs, with parameter selection based on small UAV detection requirements:

```python
CONFIG = {
    'hardware': {
        'SAMPLE_RATE': 2e6,           # 2 MHz sampling rate
        'CENTER_FREQ': 2.1e9,         # 2.1 GHz IF frequency
        'OUTPUT_FREQ': 10.25e9,       # 10.25 GHz RF output
        'CHIRP_BW': 500e6,            # 500 MHz bandwidth
        'RAMP_TIME': 500,             # 500 μs chirp duration
        'RX_GAIN': 40,                # Receiver gain setting
        'NUM_AZ_ELEMENTS': 8,         # 8-element phased array
    }
}
```

The 500 MHz bandwidth selection provides approximately 0.3-meter range resolution, which is being investigated for its suitability in small UAV detection scenarios where high spatial resolution may be beneficial for target discrimination.

### 2.3 Beamforming Implementation

The current beamforming implementation utilizes the CN0566's built-in phase control capabilities to investigate electronic beam steering:

```python
def apply_beamforming_enhanced(az_angle):
    """Beamforming implementation for azimuth scanning"""
    rf_freq = CONFIG['hardware']['OUTPUT_FREQ']
    element_spacing = 0.015  # 15mm spacing
    c = 3e8

    # Calculate phase progression for 8-element array
    k = 2 * np.pi * rf_freq / c
    phase_progression = k * element_spacing * np.sin(np.radians(az_angle)) * element_indices

    # Apply to CN0566 hardware
    my_phaser.set_beam_phase_diff(phase_diff_deg)
```

The implementation currently explores:

- ±60° azimuth coverage with variable beam positioning
- Blackman tapering for sidelobe control
- Approximately 2° beam resolution based on array geometry

### 2.4 Data Transmission Approach

The current implementation addresses the challenge of transmitting high-rate I/Q data over standard network infrastructure:

```python
def safe_chunk_data(rx_data, max_samples_per_chunk=800):
    """Data chunking for UDP transmission"""
    # Split large I/Q datasets into manageable UDP packets
    # Addresses network MTU limitations
```

Current transmission parameters under investigation:

- 50KB maximum packet size to avoid fragmentation
- 800 samples per chunk based on empirical testing
- UDP protocol selection for reduced latency (at the cost of reliability)

### 2.5 MAVLink Integration (Experimental)

An experimental MAVLink interface has been implemented to investigate adaptive radar operation based on platform motion:

```python
def mavlink_listener(udp_socket, laptop_ip, laptop_port):
    """Experimental MAVLink interface for adaptive operation"""
    # Monitors aircraft speed and GPS status
    # Investigates automatic profile switching
    # Explores scan pattern adaptation
```

This experimental feature explores:

- Automatic switching between operating profiles
- Scan pattern adaptation based on platform velocity
- Integration with standard autopilot telemetry systems

---

## 3. Signal Processing and Visualization Platform (`radar_dashboard.py`)

### 3.1 Module Overview

The laptop-based processing module implements various signal processing algorithms and visualization techniques currently under investigation for small UAV detection applications.

**Current Development Areas:**

1. Real-time FMCW signal processing
2. Target detection algorithm evaluation (CFAR methods)
3. Micro-Doppler analysis techniques
4. Multi-target tracking algorithm development
5. 3D visualization and user interface design
6. Performance optimization and system integration

### 3.2 FMCW Signal Processing Implementation

The current signal processing implementation explores various techniques for FMCW radar data analysis:

```python
def enhanced_range_doppler_processing(iq_ch0, iq_ch1, use_mti=True):
    """FMCW processing pipeline under development"""
    # 1. Reshape data into chirp matrix format
    # 2. Apply moving target indication (MTI) filtering
    # 3. Range FFT with windowing functions
    # 4. Doppler FFT for velocity analysis
    # 5. Generate range-Doppler map
```

**Current Processing Stages Under Investigation:**

- DC removal techniques (per-chirp and global methods)
- Window function evaluation (Blackman-Harris, Hamming, etc.)
- Range compression via FFT processing
- Doppler analysis for velocity estimation
- Moving Target Indication (MTI) filtering approaches

### 3.3 Target Detection Algorithm Development

The current implementation includes several Constant False Alarm Rate (CFAR) detection methods under evaluation:

```python
def advanced_cfar_detection(range_doppler_magnitude, cfar_method='average'):
    """CFAR detection methods under evaluation"""
    # Investigating: average, greatest, smallest, false_alarm methods
    # Exploring adaptive guard and reference cell sizing
    # Developing 2D CFAR for range-Doppler maps
```

**CFAR Methods Currently Under Investigation:**

- 2D CFAR processing across range-Doppler space
- Adaptive thresholding based on local noise estimation
- Comparison of different CFAR algorithms for small target detection
- False alarm rate control and performance evaluation

### 3.4 Micro-Doppler Analysis Development

An experimental micro-Doppler analysis capability is under development to investigate small UAV detection:

```python
def enhanced_drone_micro_doppler_classification(rd_map_complex_history, target_indices):
    """Experimental micro-Doppler analysis for UAV detection"""
    # 1. Investigating rotor blade flash detection (10-200 Hz)
    # 2. Exploring harmonic analysis for propeller signatures
    # 3. Developing spectral spread analysis techniques
    # 4. Researching micro-motion pattern recognition
```

**Current Research Areas:**

- Blade flash detection algorithms for periodic rotor signatures
- Harmonic analysis methods for propeller identification
- Spectral spreading analysis for micro-motion characterization
- Pattern recognition approaches for UAV classification

**Preliminary Investigation Results:**

- Initial testing suggests potential for UAV detection in controlled environments
- Classification accuracy varies significantly with range and environmental conditions
- Processing latency remains a challenge for real-time implementation

### 3.5 Multi-Target Tracking Development

A Kalman filter-based tracking system is under development to investigate multi-target tracking capabilities:

```python
class EnhancedKalmanTracker:
    """Experimental 9-state Kalman filter implementation"""
    # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    # Investigating micro-Doppler integration
    # Exploring adaptive noise models
```

**Current Tracking Development Areas:**

- 9-state extended Kalman filter implementation (position, velocity, acceleration)
- Investigation of adaptive noise models for different target types
- Experimental micro-Doppler signature integration
- Development of track quality assessment methods
- Exploration of maneuver detection algorithms

### 3.6 Object Classification Research

An experimental object classification system is being developed to investigate automated target categorization:

```python
def classify_semantic_object(cluster_df, environment_mode):
    """Experimental classification with confidence scoring"""
    # Investigating point cloud cluster analysis
    # Exploring classification categories and methods
    # Developing confidence scoring approaches
```

**Current Classification Categories Under Investigation:**

- UAV targets (with and without micro-Doppler signatures)
- Moving objects (vehicles, people, animals)
- Static objects (buildings, furniture, obstacles)
- Unknown/unclassified detections

The classification accuracy and reliability are currently being evaluated under various environmental conditions.

---

## 4. User Interface and Visualization Development

### 4.1 Dashboard Design Approach

The current user interface implementation utilizes a web-based dashboard to provide real-time visualization and system control capabilities. The interface design focuses on presenting complex radar data in an accessible format for research and educational purposes.

### 4.2 3D Visualization Implementation

The primary visualization component presents radar data in a 3D point cloud format:

**Current Visualization Elements:**

- Radar field-of-view representation (semi-transparent cone)
- Distance reference markers (concentric range rings)
- Active beam direction indicator
- Detection points with configurable color coding
- Experimental voxel-based object representation
- Tracking visualization (bounding boxes and velocity vectors)

**Color Coding Scheme Under Development:**

- Red: Potential UAV detections
- Green: Moving object detections
- Orange: Static object detections
- Cyan: Structural/environmental elements
- Purple: Unclassified detections

The effectiveness of different visualization approaches is currently being evaluated for various use cases.

### 4.3 Control Interface Development

The control interface provides access to various system parameters and visualization options:

**System Monitoring Elements:**

- Network connectivity status indicators
- Scan progress and completion tracking
- Operating mode display (indoor/outdoor profiles)

**Parameter Control Interface:**

- Data source selection (live data vs. synthetic test data)
- Environmental profile selection for algorithm adaptation
- Visualization color scheme selection (velocity, range, signal strength)
- Display element toggles for different visualization components
- Performance mode selection for computational load management

**Adjustable Parameters:**

- Point size for visualization clarity
- Voxel resolution for spatial quantization (0.1-1.0m range)
- Object representation sizing
- Signal-to-noise ratio thresholds (-120 to -40 dB range)

The interface design continues to evolve based on user feedback and operational requirements.

### 4.4 Diagnostic Visualization Tools

Multiple diagnostic views have been implemented to support algorithm development and system analysis:

**Current Diagnostic Displays:**

- Range-Doppler heatmap visualization (velocity vs. distance)
- Elevation angle distribution analysis
- Top-down spatial view (bird's-eye perspective)
- Experimental micro-Doppler signature display
- Range profile history (waterfall format)

These diagnostic tools facilitate algorithm development by providing multiple perspectives on the radar data and enabling detailed analysis of system performance.

### 4.5 Information Display Panels

Several information panels provide real-time system status and analysis results:

**Experimental Micro-Doppler Analysis Display:**

- Blade frequency estimation (when available)
- Detection confidence metrics
- Signal strength indicators
- Classification results (preliminary)

**Object Tracking Information:**

- Tracked object identifiers and classifications
- Tracking status (preliminary/confirmed)
- Associated detection counts
- Range and velocity estimates

**System Status Monitoring:**

- Current detection counts
- Active beam direction
- Tracked object counts
- Scan completion status

**Performance Monitoring:**

- Processing latency measurements
- CPU and memory utilization
- Current performance optimization level

These displays support both system operation and algorithm development by providing insight into system performance and processing results.

---

## 5. Signal Processing Algorithm Development

### 5.1 Multipath Mitigation Research

An experimental multipath suppression algorithm is under development:

```python
def coherent_multipath_suppression(range_doppler_history, coherence_threshold=0.8):
    """Experimental multipath suppression using multi-frame coherence"""
    # Investigating phase consistency analysis across frames
    # Exploring incoherent multipath attenuation
    # Developing direct-path signature preservation
```

### 5.2 Moving Target Indication (MTI) Development

Multiple MTI processing approaches are being investigated:

```python
def enhanced_mti_processing(radar_data_history, mti_mode='3pulse'):
    """MTI processing methods under investigation"""
    # Exploring 2-pulse and 3-pulse cancellation
    # Investigating static clutter removal
    # Developing moving target preservation techniques
```

### 5.3 Temporal Filtering Research

Temporal consistency filtering is being explored to improve detection reliability:

```python
def apply_temporal_coherence_filter(frame_history):
    """Experimental temporal consistency filtering"""
    # Investigating spatial grid analysis across frames
    # Exploring false alarm reduction techniques
    # Developing persistent target track maintenance
```

These algorithms are currently in various stages of development and testing, with performance evaluation ongoing.

---

## 6. Current System Performance and Limitations

### 6.1 Measured Performance Characteristics

**Range Performance (Preliminary Results):**

- Minimum detectable range: ~0.5 meters
- Maximum tested range: 25 meters (indoor), 50 meters (outdoor, limited testing)
- Theoretical range resolution: 0.3 meters (based on 500 MHz bandwidth)
- Range accuracy: Under evaluation

**Angular Performance (Based on Array Geometry):**

- Azimuth coverage: ±60° (theoretical)
- Estimated azimuth resolution: ~2° (based on 8-element array)
- Elevation coverage: ±30° (estimated)
- Angular accuracy: Currently under evaluation

**Velocity Measurement Capabilities:**

- Theoretical velocity range: ±15 m/s (based on PRF and wavelength)
- Velocity resolution: ~0.1 m/s (theoretical)
- Velocity accuracy: Under investigation

### 6.2 System Performance Observations

**Processing Performance (Current Implementation):**

- Update rate: ~10 Hz (varies with processing load)
- Scan rate: Variable (15-25 angles/second depending on mode)
- Processing latency: 100-200ms (varies with algorithm complexity)
- End-to-end latency: Under measurement

**Network Performance (Observed):**

- Sustained data rate: ~2 MB/s (varies with network conditions)
- Packet loss: <5% over WiFi (varies with network quality)
- Network latency: 5-20ms (typical for local network)

---

## 7. UAV Detection Research Progress

### 7.1 Micro-Doppler Investigation

The current research investigates micro-Doppler signatures for small UAV detection:

**Theoretical Detection Characteristics Under Investigation:**

- Blade flash frequency range: 20-200 Hz (varies with rotor RPM)
- Harmonic content analysis for propeller identification
- Spectral spreading due to micro-motion effects
- Temporal pattern recognition for periodic signatures

**UAV Types Under Investigation:**

- Quadcopter configurations (4-rotor systems)
- Hexacopter configurations (6-rotor systems)
- Octocopter configurations (8-rotor systems)
- Small fixed-wing UAVs with propellers

### 7.2 Preliminary Detection Results

**Current Performance Assessment (Limited Testing):**

- Detection capability varies significantly with range and environmental conditions
- False alarm rates remain high in current implementation
- Classification accuracy requires further development
- Minimum detectable target size is under investigation

**Identified Challenges:**

- Distinguishing UAV signatures from environmental clutter
- Maintaining consistent detection performance across different ranges
- Reducing false alarm rates while preserving detection sensitivity
- Real-time processing constraints limit algorithm complexity

---

## 8. System Integration and Interoperability

### 8.1 MAVLink Protocol Integration (Experimental)

An experimental MAVLink interface has been implemented to explore integration with autopilot systems:

```python
def mavlink_exporter_thread():
    """Experimental MAVLink integration for obstacle reporting"""
    # Investigating connection to autopilot systems
    # Exploring OBSTACLE_DISTANCE message transmission
    # Researching autonomous collision avoidance integration
```

**Protocol Support Under Investigation:**

- MAVLink v2.0 message format compatibility
- ArduPilot integration possibilities
- PX4 autopilot communication protocols

### 8.2 Network Architecture Considerations

The current implementation has been tested with several network configurations:

**Network Configurations Tested:**

- Standard WiFi networks (802.11n/ac)
- Direct Ethernet connections (reduced latency)
- Pi-to-laptop hotspot communication (isolated network)

Network performance varies significantly across different configurations, with Ethernet providing the most reliable data transmission.

---

## 9. Educational and Research Applications

### 9.1 Educational Value

This development project provides several educational opportunities:

1. **Signal Processing Implementation**: Practical application of digital signal processing concepts
2. **Radar Systems Engineering**: Hands-on experience with phased array radar systems
3. **Algorithm Development**: Implementation and evaluation of detection and tracking algorithms
4. **Real-time Systems Design**: Investigation of low-latency processing requirements
5. **Distributed System Architecture**: Network-based processing system development

### 9.2 Research Platform Capabilities

The current platform enables investigation of several research areas:

1. **Detection Algorithm Development**: CFAR methods, machine learning approaches
2. **Tracking Algorithm Research**: Multi-target tracking, trajectory prediction
3. **Sensor Fusion Studies**: Integration with complementary sensor systems
4. **Autonomous System Integration**: Robotic platform integration research
5. **Counter-UAS Technology**: Small UAV detection and classification research

The modular architecture facilitates experimentation with different algorithms and approaches while maintaining a stable hardware platform.

---

## 10. Current Development Status and Contributions

### 10.1 Implementation Progress

The current development has achieved several milestones:

1. **Micro-Doppler Processing Development**: Initial algorithms for small UAV signature analysis
2. **3D Visualization Implementation**: Interactive point cloud display with multiple view modes
3. **Performance Management System**: Adaptive processing based on available system resources
4. **Distributed Architecture**: Functional Pi-laptop processing distribution
5. **Protocol Integration**: Experimental MAVLink interface development

### 10.2 Algorithm Development Areas

Several algorithms are currently under development and evaluation:

1. **Multipath Suppression**: Phase-based approaches to multipath mitigation
2. **Extended Kalman Tracking**: 9-state filter with experimental micro-Doppler integration
3. **Object Classification**: Voxel-based categorization methods
4. **Temporal Filtering**: Multi-frame consistency analysis for false alarm reduction

These algorithms represent ongoing research efforts with varying levels of maturity and validation.

---

## 11. Future Development Directions

### 11.1 Planned Algorithm Improvements

Several areas have been identified for future development:

1. **Machine Learning Integration**: Investigation of deep learning approaches for improved classification
2. **Multi-Static Operation**: Exploration of distributed radar network capabilities
3. **Advanced Tracking Methods**: Research into particle filters and interacting multiple model (IMM) approaches
4. **Sensor Fusion**: Investigation of integration with optical and LiDAR sensors
5. **Cloud Processing**: Exploration of remote processing capabilities for computationally intensive algorithms

### 11.2 Research Extension Opportunities

The current platform provides a foundation for several research extensions:

1. **Multi-UAV Detection**: Investigation of swarm detection and tracking capabilities
2. **Behavioral Analysis**: Research into motion pattern recognition and intent assessment
3. **Environmental Adaptation**: Development of automatic parameter optimization methods
4. **Stealth UAV Detection**: Investigation of detection methods for low-observable targets
5. **Airspace Integration**: Research into integration with air traffic management systems

These research directions represent potential future work that could build upon the current development efforts.

---

## 12. Conclusions and Current Status

### 12.1 Development Summary

This project represents an ongoing investigation into the application of commercial radar development platforms for small UAV detection research. The current implementation builds upon established CN0566 reference designs while exploring various signal processing and visualization approaches for UAV detection applications.

**Current System Capabilities:**

- Functional distributed processing architecture
- Real-time data acquisition and transmission
- Multiple signal processing algorithm implementations (in various stages of development)
- Interactive 3D visualization with configurable display options
- Experimental micro-Doppler analysis capabilities
- Multi-target tracking algorithm development
- Web-based user interface for research and educational applications

### 12.2 Identified Limitations and Challenges

Several limitations have been identified during development:

- Detection performance varies significantly with environmental conditions
- False alarm rates remain higher than desired in current implementations
- Real-time processing constraints limit algorithm complexity
- Network transmission reliability affects system performance
- Classification accuracy requires further development and validation

### 12.3 Educational and Research Value

Despite current limitations, the platform provides significant value for educational and research purposes:

- Practical implementation of radar signal processing concepts
- Investigation of distributed system architectures
- Algorithm development and evaluation platform
- Real-time system design challenges and solutions
- Integration of multiple technologies (radar, networking, visualization)

---

## 13. Technical Specifications (Current Implementation)

| Parameter                           | Current Status                            |
| ----------------------------------- | ----------------------------------------- |
| **Operating Frequency**             | 10.25 GHz (X-band)                        |
| **Signal Bandwidth**                | 500 MHz                                   |
| **Tested Range**                    | 0.5-25m (indoor), limited outdoor testing |
| **Theoretical Range Resolution**    | 0.3 meters                                |
| **Angular Coverage**                | ±60° azimuth (theoretical)                |
| **Estimated Angular Resolution**    | ~2° (based on array geometry)             |
| **Theoretical Velocity Resolution** | 0.1 m/s                                   |
| **Current Update Rate**             | ~10 Hz (varies with processing load)      |
| **Processing Platform**             | Raspberry Pi 4 + Laptop PC                |
| **User Interface**                  | Web-based dashboard with 3D visualization |

### 13.1 Acknowledgments

This work builds upon the foundational CN0566 implementations developed by Jon Kraft and the Analog Devices team. Their comprehensive reference designs and documentation provided the essential hardware control and signal processing foundation that enabled this research investigation.

The current implementation represents work in progress, with ongoing development focused on improving detection performance, reducing false alarm rates, and expanding the system's research capabilities for small UAV detection applications.
