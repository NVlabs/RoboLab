# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Trajectory smoothness and quality metrics.
"""

import numpy as np


def compute_joint_isj_from_velocity(joint_velocity: np.ndarray, dt: float) -> float:
    """
    Compute Integrated Squared Jerk (ISJ) in joint space, summed over all joints.
    Starts from joint velocity data.

    Args:
        joint_velocity: Array of shape (T, n_joints) containing joint velocities over time
        dt: Time step between samples (seconds)

    Returns:
        ISJ: Integrated squared jerk (scalar), units: rad^2/s^5
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    joint_velocity = np.asarray(joint_velocity, dtype=np.float64)

    # Compute derivatives using central differences
    acceleration = np.gradient(joint_velocity, dt, axis=0)  # (T, n_joints)
    jerk = np.gradient(acceleration, dt, axis=0)  # (T, n_joints)

    # Squared jerk summed over joints at each timestep
    squared_jerk_sum = np.sum(jerk**2, axis=1)  # (T,)

    # Integrate over time using trapezoidal rule
    isj = np.trapz(squared_jerk_sum, dx=dt)

    return isj


def compute_joint_isj_per_joint_from_velocity(
    joint_velocity: np.ndarray, dt: float
) -> np.ndarray:
    """
    Compute ISJ for each joint separately, starting from velocity.

    Args:
        joint_velocity: Array of shape (T, n_joints)
        dt: Time step (seconds)

    Returns:
        isj_per_joint: Array of shape (n_joints,) with ISJ for each joint
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    joint_velocity = np.asarray(joint_velocity, dtype=np.float64)

    acceleration = np.gradient(joint_velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    # Integrate squared jerk for each joint
    isj_per_joint = np.trapz(jerk**2, dx=dt, axis=0)  # (n_joints,)

    return isj_per_joint


def compute_joint_isj_from_position(joint_positions: np.ndarray, dt: float) -> float:
    """
    Compute Integrated Squared Jerk (ISJ) in joint space, summed over all joints.
    Uses central differences via np.gradient for better numerical accuracy.

    Args:
        joint_positions: Array of shape (T, n_joints) containing joint positions over time
        dt: Time step between samples (seconds)

    Returns:
        ISJ: Integrated squared jerk (scalar), units: rad^2/s^5
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    joint_positions = np.asarray(joint_positions, dtype=np.float64)

    # Compute derivatives using central differences
    velocity = np.gradient(joint_positions, dt, axis=0)  # (T, n_joints)
    acceleration = np.gradient(velocity, dt, axis=0)  # (T, n_joints)
    jerk = np.gradient(acceleration, dt, axis=0)  # (T, n_joints)

    # Squared jerk summed over joints at each timestep
    squared_jerk_sum = np.sum(jerk**2, axis=1)  # (T,)

    # Integrate over time using trapezoidal rule
    isj = np.trapz(squared_jerk_sum, dx=dt)

    return isj


def compute_joint_isj_per_joint_from_position(joint_positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute ISJ for each joint separately.

    Args:
        joint_positions: Array of shape (T, n_joints)
        dt: Time step (seconds)

    Returns:
        isj_per_joint: Array of shape (n_joints,) with ISJ for each joint
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    joint_positions = np.asarray(joint_positions, dtype=np.float64)

    velocity = np.gradient(joint_positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    # Integrate squared jerk for each joint
    isj_per_joint = np.trapz(jerk**2, dx=dt, axis=0)  # (n_joints,)

    return isj_per_joint


# =============================================================================
# End-Effector (Cartesian) Space ISJ
# =============================================================================


def compute_ee_isj_from_position(ee_position: np.ndarray, dt: float) -> float:
    """
    Compute Integrated Squared Jerk (ISJ) for end-effector position in Cartesian space.

    Args:
        ee_position: Array of shape (T, 3) containing EE positions [x, y, z] over time
        dt: Time step between samples (seconds)

    Returns:
        ISJ: Integrated squared jerk (scalar), units: m^2/s^5
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    ee_position = np.asarray(ee_position, dtype=np.float64)

    # Compute derivatives using central differences
    velocity = np.gradient(ee_position, dt, axis=0)  # (T, 3)
    acceleration = np.gradient(velocity, dt, axis=0)  # (T, 3)
    jerk = np.gradient(acceleration, dt, axis=0)  # (T, 3)

    # Squared jerk magnitude at each timestep: ||jerk||^2 = jerk_x^2 + jerk_y^2 + jerk_z^2
    squared_jerk_magnitude = np.sum(jerk**2, axis=1)  # (T,)

    # Integrate over time using trapezoidal rule
    isj = np.trapz(squared_jerk_magnitude, dx=dt)

    return isj


def compute_ee_isj_from_velocity(ee_velocity: np.ndarray, dt: float) -> float:
    """
    Compute Integrated Squared Jerk (ISJ) for end-effector in Cartesian space.
    Starts from EE velocity data.

    Args:
        ee_velocity: Array of shape (T, 3) containing EE velocities [vx, vy, vz] over time
        dt: Time step between samples (seconds)

    Returns:
        ISJ: Integrated squared jerk (scalar), units: m^2/s^5
    """
    # Cast to float64 to avoid overflow when squaring jerk values
    ee_velocity = np.asarray(ee_velocity, dtype=np.float64)

    # Compute derivatives using central differences
    acceleration = np.gradient(ee_velocity, dt, axis=0)  # (T, 3)
    jerk = np.gradient(acceleration, dt, axis=0)  # (T, 3)

    # Squared jerk magnitude at each timestep
    squared_jerk_magnitude = np.sum(jerk**2, axis=1)  # (T,)

    # Integrate over time using trapezoidal rule
    isj = np.trapz(squared_jerk_magnitude, dx=dt)

    return isj


# =============================================================================
# Path Length
# =============================================================================


def compute_ee_path_length(ee_position: np.ndarray) -> float:
    """
    Compute the total path length of the end-effector trajectory in Cartesian space.

    Path length is the sum of Euclidean distances between consecutive points:
        L = Σ ||p_{k+1} - p_k||

    Args:
        ee_position: Array of shape (T, 3) containing EE positions [x, y, z] over time

    Returns:
        path_length: Total path length (scalar), units: meters
    """
    # Cast to float64 to avoid overflow in norm computation
    ee_position = np.asarray(ee_position, dtype=np.float64)

    # Compute displacement vectors between consecutive points
    displacements = np.diff(ee_position, axis=0)  # (T-1, 3)

    # Compute Euclidean distance for each segment
    segment_lengths = np.linalg.norm(displacements, axis=1)  # (T-1,)

    # Sum all segment lengths
    path_length = np.sum(segment_lengths)

    return path_length


# =============================================================================
# SPARC (Spectral Arc Length)
# =============================================================================


def compute_sparc(
    speed: np.ndarray,
    dt: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amplitude_threshold: float = 0.05,
) -> float:
    """
    Compute Spectral Arc Length (SPARC) smoothness metric.

    SPARC measures the arc length of the Fourier magnitude spectrum of the speed
    profile. More negative values indicate smoother movements.

    Reference:
        Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012).
        A robust and sensitive metric for quantifying movement smoothness.
        IEEE transactions on biomedical engineering, 59(8), 2126-2136.

    Args:
        speed: Speed profile, array of shape (T,). This should be the magnitude
               of velocity (e.g., ||v|| for Cartesian, or can be per-joint).
        dt: Time step between samples (seconds)
        padlevel: Amount of zero-padding for FFT (multiplier of signal length)
        fc: Maximum cutoff frequency for arc length calculation (Hz)
        amplitude_threshold: Threshold for determining adaptive cutoff frequency,
                             as a fraction of the maximum amplitude

    Returns:
        sparc: SPARC value (negative scalar). More negative = smoother.
    """
    # Number of samples
    N = len(speed)

    # Zero-pad the signal
    nfft = int(2 ** np.ceil(np.log2(N)) * padlevel)

    # Compute FFT and normalize
    speed_fft = np.fft.rfft(speed, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=dt)

    # Magnitude spectrum, normalized by max
    magnitude = np.abs(speed_fft)
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    # Find adaptive cutoff frequency based on amplitude threshold
    # Use the frequency where the amplitude drops below threshold, or fc, whichever is smaller
    above_threshold = magnitude >= amplitude_threshold
    if np.any(above_threshold):
        # Find the highest frequency index above threshold
        last_idx = np.max(np.where(above_threshold)[0])
        fc_adaptive = min(freq[last_idx], fc)
    else:
        fc_adaptive = fc

    # Select frequencies up to the cutoff
    freq_mask = freq <= fc_adaptive
    freq_sel = freq[freq_mask]
    magnitude_sel = magnitude[freq_mask]

    # Compute arc length of the spectrum
    # SPARC = -∫ sqrt((1/fc)^2 + (dV/dω)^2) dω
    if len(freq_sel) < 2:
        return 0.0

    # Numerical derivative of magnitude spectrum
    d_magnitude = np.diff(magnitude_sel)
    d_freq = np.diff(freq_sel)

    # Arc length integral
    # Each segment contributes sqrt((dω/fc)^2 + dV^2)
    arc_length_elements = np.sqrt((d_freq / fc_adaptive) ** 2 + d_magnitude**2)
    arc_length = np.sum(arc_length_elements)

    # SPARC is negative arc length (so more negative = longer arc = less smooth)
    sparc = -arc_length

    return sparc


def compute_sparc_from_velocity(velocity: np.ndarray, dt: float, **kwargs) -> float:
    """
    Compute SPARC from a velocity trajectory (multi-dimensional).

    Computes speed as the L2 norm of velocity, then applies SPARC.

    Args:
        velocity: Velocity array of shape (T,) or (T, n_dims)
        dt: Time step (seconds)
        **kwargs: Additional arguments passed to compute_sparc

    Returns:
        sparc: SPARC value (negative scalar)
    """
    if velocity.ndim == 1:
        speed = np.abs(velocity)
    else:
        speed = np.linalg.norm(velocity, axis=1)

    return compute_sparc(speed, dt, **kwargs)


def compute_sparc_per_joint(
    joint_velocity: np.ndarray, dt: float, **kwargs
) -> np.ndarray:
    """
    Compute SPARC for each joint separately.

    Args:
        joint_velocity: Array of shape (T, n_joints)
        dt: Time step (seconds)
        **kwargs: Additional arguments passed to compute_sparc

    Returns:
        sparc_per_joint: Array of shape (n_joints,) with SPARC for each joint
    """
    n_joints = joint_velocity.shape[1]
    sparc_values = np.zeros(n_joints)

    for i in range(n_joints):
        # Use absolute velocity as "speed" for each joint
        speed = np.abs(joint_velocity[:, i])
        sparc_values[i] = compute_sparc(speed, dt, **kwargs)

    return sparc_values


def compute_ee_sparc_from_position(ee_position: np.ndarray, dt: float, **kwargs) -> float:
    """
    Compute SPARC for end-effector trajectory from position data.

    Computes velocity via numerical differentiation, then speed as ||v||,
    then applies SPARC.

    Args:
        ee_position: Array of shape (T, 3) containing EE positions [x, y, z]
        dt: Time step (seconds)
        **kwargs: Additional arguments passed to compute_sparc

    Returns:
        sparc: SPARC value (negative scalar). More negative = less smooth.
    """
    # Compute velocity using central differences
    velocity = np.gradient(ee_position, dt, axis=0)  # (T, 3)

    # Compute speed (magnitude of velocity)
    speed = np.linalg.norm(velocity, axis=1)  # (T,)

    return compute_sparc(speed, dt, **kwargs)


def compute_ee_sparc_from_velocity(ee_velocity: np.ndarray, dt: float, **kwargs) -> float:
    """
    Compute SPARC for end-effector trajectory from velocity data.

    Args:
        ee_velocity: Array of shape (T, 3) containing EE velocities [vx, vy, vz]
        dt: Time step (seconds)
        **kwargs: Additional arguments passed to compute_sparc

    Returns:
        sparc: SPARC value (negative scalar). More negative = less smooth.
    """
    # Compute speed (magnitude of velocity)
    speed = np.linalg.norm(ee_velocity, axis=1)  # (T,)

    return compute_sparc(speed, dt, **kwargs)
