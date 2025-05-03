"""Helper functions for sequence generation.

Additionally: Global definition of system settings to be imported by sequence constructors.
"""

import math
import time
from pathlib import Path

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
from pypulseq import Sequence
from pypulseq import SigpyPulseOpts
from pypulseq import make_gauss_pulse
from pypulseq import make_sigpy_pulse
from pypulseq import make_sinc_pulse
from pypulseq.opts import Opts

from utilities.type import Trajectory

GAMMA = 42.576e6
PERFORMANCE = 1  # max: 50, normal: 0.75*50 silent: 0.5*50

# System specifications for OSI one scanner
system = Opts(
    rf_dead_time=20e-6,  # dead time at the beginning of RF event, covered by rf_delay.
    rf_ringdown_time=30e-6,  # Time delay at the end of an RF event
    # Set raster times to spectrum card frequency (timing checks)
    grad_raster_time=1e-6,
    rf_raster_time=1e-6,
    block_duration_raster=1e-6,
    adc_raster_time=1e-6,
    # Set maximum gradient amplitude
    max_grad=PERFORMANCE * 250e3 / GAMMA * 1e3,  # ~5.9 mT/m
    grad_unit='mT/m',
    # Set maximum slew rate
    max_slew=PERFORMANCE * 100,
    slew_unit='T/m/s',
    B0=50e-3,
    gamma=GAMMA,
)


# Helper function to round values to raster time
def raster(val: float, precision: float) -> float:
    """Fit value to gradient raster.

    Parameters
    ----------
    val
        Time value to be aligned on the raster.
    precision
        Raster precision, e.g. system.grad_raster_time or system.adc_raster_time

    Returns
    -------
        Value with given time/raster precision
    """
    gridded_val = np.round(val / precision) * precision
    return gridded_val


# Maps the FOV and encoding parameters to the respective dictionaries
def map_fov_enc(channels, input_fov, input_enc) -> tuple[dict, dict]:
    """
    Map the field of view (FOV) and encoding parameters from the input objects to dictionaries.

    Parameters
    ----------
        channels: An object containing the channel names for readout (ro),
        phase encoding 1 (pe1), and phase encoding 2 (pe2).
        input_fov: An object containing the field of view parameters.
        input_enc: An object containing the encoding parameters.

    Returns
    -------
        tuple[dict, dict]: A tuple containing two dictionaries:
            - n_enc: A dictionary with encoding parameters for 'ro', 'pe1', and 'pe2'.
            - fov: A dictionary with field of view parameters for 'ro', 'pe1', and 'pe2'.
    """
    n_enc = {}
    fov = {}
    n_enc['ro'] = getattr(input_enc, channels.ro)
    n_enc['pe1'] = getattr(input_enc, channels.pe1)
    n_enc['pe2'] = getattr(input_enc, channels.pe2)

    fov['ro'] = getattr(input_fov, channels.ro)
    fov['pe1'] = getattr(input_fov, channels.pe1)
    fov['pe2'] = getattr(input_fov, channels.pe2)

    return (n_enc, fov)


# Returns the trajectory of kspace
def get_traj(
    n_enc: dict,
    etl: int,
    trajectory: Trajectory = Trajectory.INOUT,
) -> list:
    """
    Generate phase encoding trajectory for MRI sequences.

    Parameters
    ----------
    n_enc (dict): Dictionary containing the number of encoding steps for 'pe1' and 'pe2'.
    etl (int): Echo train length.
    trajectory (Trajectory): Type of trajectory to generate.
        - Trajectory.INOUT: Center-out trajectory.
        - Trajectory.OUTIN: Outside-in trajectory.
        - Trajectory.LINEAR: Linear trajectory.
        - Trajectory.ASCENDING: Ascending phase encoding trajectory.

    Returns
    -------
    list: List of phase encoding points based on the specified trajectory.
    """
    # Calculate center out trajectory
    # Calculate gradient areas for phase encoding directions
    pe1 = np.arange(n_enc['pe1']) - n_enc['pe1'] / 2
    if n_enc['pe2'] == 1:  # exception if only 1 PE2 step is present
        pe2 = np.array([0])
    else:
        pe2 = np.arange(n_enc['pe2']) - (n_enc['pe2']) / 2

    pe_points = np.stack([grid.flatten() for grid in np.meshgrid(pe1, pe2)], axis=-1)

    pe_mag = np.sum(np.square(pe_points), axis=-1)  # calculate magnitude of all gradient combinations
    pe_mag_sorted = np.argsort(pe_mag)

    if trajectory is Trajectory.INOUT:
        pe_traj = pe_points[pe_mag_sorted, :]  # sort the points based on magnitude

    elif trajectory is Trajectory.OUTIN:
        pe_mag_sorted = np.flip(pe_mag_sorted)
        pe_traj = pe_points[pe_mag_sorted, :]  # sort the points based on magnitude

    elif trajectory is Trajectory.LINEAR:
        center_pos = 1 / 2  # where the center of kspace should be in the echo train
        num_points = np.size(pe_mag_sorted)
        linear_pos = np.zeros(num_points, dtype=int) - 10
        center_point = int(np.round(np.size(pe_mag) * center_pos))
        odd_indices = 1
        even_indices = 1
        linear_pos[center_point] = pe_mag_sorted[0]

        for idx in range(1, num_points):
            # check if its in bounds first
            if center_point - (idx + 1) / 2 >= 0 and idx % 2:
                k_idx = center_point - odd_indices
                odd_indices += 1
            elif (center_point + idx / 2 < num_points and idx % 2 == 0) or (
                center_point - (idx + 1) / 2 < 0 and idx % 2
            ):
                k_idx = center_point + even_indices
                even_indices += 1
            elif center_point + idx / 2 >= num_points and idx % 2 == 0:
                k_idx = center_point - odd_indices
                odd_indices += 1
            else:
                print('Sorting error')
            linear_pos[k_idx] = pe_mag_sorted[idx]

        pe_traj = pe_points[linear_pos, :]  # sort the points based on magnitude

    elif (
        trajectory is Trajectory.ASCENDING
    ):  # ascending phase encoding, from negative frequencies to positive frequencies for each pe2 step
        assert n_enc['pe1'] % etl == 0
        # PE dir 1
        pe_traj = pe1

    return pe_traj


# Returns the echo trains for given trajectory and number of echoes per excitation
def get_trains(
    pe_traj: list,
    n_enc: dict,
    fov: dict,
    etl: int,
    trajectory: Trajectory,
) -> tuple[list, list]:
    """
    Divides phase encoding (PE) steps into echo trains based on the given trajectory.

    Parameters
    ----------
    pe_traj (list): List of phase encoding trajectory points.
    n_enc (dict): Dictionary containing the number of phase encoding steps for each direction (keys: 'pe1', 'pe2').
    fov (dict): Dictionary containing the field of view for each direction (keys: 'pe1', 'pe2').
    etl (int): Echo train length.
    trajectory (Trajectory): Trajectory object containing the name of the trajectory.

    Returns
    -------
    tuple[list, list]: A tuple containing two lists:
        - trains: List of echo trains with normalized gradient areas for each k-point.
        - trains_pos: List of echo trains with positions in the phase encoding space.
    """
    trains = []
    trains_pos = []

    # Divide all PE steps into echo trains
    if trajectory.name == 'ASCENDING':
        num_trains = int(
            np.ceil(n_enc['pe1'] / etl)
        )  # due to "slice wise" acquisition, only first pe direction is divided into trains
        temp = [pe_traj[k::num_trains] for k in range(num_trains)]
        for k in (
            np.arange(n_enc['pe2']) - n_enc['pe2'] / 2
        ):  # add the second pe direction to the trains, but keep the order of pe1 steps in each train
            for i in np.arange(num_trains):
                for j in np.arange(etl):
                    trains.append([temp[i][j] / fov['pe1'], k / fov['pe2']])
                    trains_pos.append([temp[i][j] + n_enc['pe1'] / 2, k + n_enc['pe2'] / 2])

        trains = [trains[k * etl : (k + 1) * etl] for k in range(num_trains * n_enc['pe2'])]
        trains_pos = [trains_pos[k * etl : (k + 1) * etl] for k in range(num_trains * n_enc['pe2'])]
    else:
        if n_enc['pe2'] == 1:  # exception if only 1 PE2 step is present
            pe_order = np.array([[pe_traj[k, 0] + n_enc['pe1'] / 2, 0] for k in range(len(pe_traj))])
        else:
            pe_order = np.array(
                [[pe_traj[k, 0] + n_enc['pe1'] / 2, pe_traj[k, 1] + n_enc['pe2'] / 2] for k in range(len(pe_traj))]
            )
        # correct pe order to integer steps starting at 0
        if (min(pe_order[:, 0]) > 0) or (min(pe_order[:, 1]) > 0):
            pe_order = np.floor(pe_order)

        num_trains = int(np.ceil(n_enc['pe1'] * n_enc['pe2'] / etl))  # both pe directions are divided into trains
        pe_traj[:, 0] /= fov['pe1']  # calculate the required gradient area for each k-point
        pe_traj[:, 1] /= fov['pe2']
        trains = [pe_traj[k::num_trains, :] for k in range(num_trains)]
        trains_pos = [pe_order[k::num_trains, :] for k in range(num_trains)]

    return trains, trains_pos


def get_esp_etl(
    trains: list,
    tau_1: float,
    tau_2: float,
    tau_3: float,
    te: float,
    T2: int = 100,
) -> dict:
    """Get min TE, effective TE, max ETL.

    Calculate the minimum TE or echo spacing, effective echo time and maximum echo train length (ETL)
    assuming a threshold of 20% of given T2 value for the maximum encoding time.

    Parameters
    ----------
    trains (list): List of echo trains for each k-point.
    tau_1 (float): First time constant in seconds.
    tau_2 (float): Second time constant in seconds.
    tau_3 (float): Third time constant in seconds.
    te (float): Echo time in seconds.
    T2 (int): T2 value of the main tissue of interest in milliseconds.
    """
    # minimum echo spacing to accommodate gradients
    min_te = -(min(tau_1, tau_2, tau_3) - te / 2) * 2
    min_te = np.ceil(min_te * 1e3) * 1e-3  # round to 1 ms -> new echo time

    # sampling duration [ms] till signal drops to 20%
    max_sampling = -math.log(0.2) * T2 * 1e-3

    # maximum numbers of 180° echoes fitting in sampling duration
    max_etl = int(np.floor(max_sampling / min_te))

    # get effective echo time from trains
    # calculate distance to k-space center for each point
    abs_trains = [[(pe1**2 + pe2**2) ** 0.5 for pe1, pe2 in train] for train in trains]
    # find smallest distance to k-space center in each train
    min_val = [min(train) for train in abs_trains]
    # in which trains is the smallest distance sampled, multiple trains possible
    which_train = [i for i, x in enumerate(min_val) if x == min(min_val)]
    # get echo number in each train for each smallest distance, multiple echoes possible
    which_echo = []
    for i in which_train:
        which_echo.append([x + 1 for x, y in enumerate(abs_trains[i]) if y == min_val[i]])

    # get effective echo time in ms
    eff_te = [[echo * te * 1e3 for echo in i] for i in which_echo]

    return {
        'min_te': min_te,
        'max_etl': max_etl,
        'center_train': which_train,
        'center_echo': which_echo,
        'effective_te': eff_te,
    }


def create_ismrmd_header(n_enc: dict, fov: dict, system: Opts) -> list:
    """
    Create an ISMRMD header.

    Parameters
    ----------
    n_enc (dict): A dictionary containing the encoding matrix size with keys 'ro', 'pe1', and 'pe2'.
    fov (dict): A dictionary containing the field of view with keys 'ro', 'pe1', and 'pe2'.
    system (Opts): Contains system parameters including B0 and gamma .

    Returns
    -------
    list: An ISMRMD header object.
    """
    # Create ISMRMRD header
    header = ismrmrd.xsd.ismrmrdHeader()

    # experimental conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = system.B0 * system.gamma / (2 * np.pi)
    header.experimentalConditions = exp

    # set fov and matrix size
    efov = ismrmrd.xsd.fieldOfViewMm()  # kspace fov in mm
    efov.x = fov['ro'] * 1e3
    efov.y = fov['pe1'] * 1e3
    efov.z = fov['pe2'] * 1e3

    rfov = ismrmrd.xsd.fieldOfViewMm()  # image fov in mm
    rfov.x = fov['ro'] * 1e3
    rfov.y = fov['pe1'] * 1e3
    rfov.z = fov['pe2'] * 1e3

    ematrix = ismrmrd.xsd.matrixSizeType()  # encoding dimensions
    ematrix.x = n_enc['ro']
    ematrix.y = n_enc['pe1']
    ematrix.z = n_enc['pe2']

    rmatrix = ismrmrd.xsd.matrixSizeType()  # image dimensions
    rmatrix.x = n_enc['ro']
    rmatrix.y = n_enc['pe1']
    rmatrix.z = n_enc['pe2']

    # set encoded and recon spaces
    escape = ismrmrd.xsd.encodingSpaceType()
    escape.matrixSize = ematrix
    escape.fieldOfView_mm = efov

    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov

    # encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()

    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_1.minimum = 0
    limits.kspace_encoding_step_1.maximum = n_enc['pe1'] - 1
    limits.kspace_encoding_step_1.center = int(n_enc['pe1'] / 2)

    limits.kspace_encoding_step_2 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_2.minimum = 0
    limits.kspace_encoding_step_2.maximum = n_enc['pe2'] - 1
    limits.kspace_encoding_step_2.center = int(n_enc['pe2'] / 2)

    # encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.encodedSpace = escape
    encoding.reconSpace = rspace
    # Trajectory type required by gadgetron (not by mrpro)
    encoding.trajectory = ismrmrd.xsd.trajectoryType('cartesian')

    encoding.encodingLimits = limits

    header.encoding.append(encoding)

    return header


def check_seq_timing(seq: Sequence):
    """
    Check the timing of a given sequence and prints the results.

    Parameters
    ----------
    seq (Sequence): The sequence object to be checked.
    The function calls the `check_timing` method of the sequence object, which returns a tuple containing a boolean
    indicating whether the timing check passed and a list of error messages if it failed. If the timing check passes,
    it prints a success message and the duration of the sequence. If the timing check fails, it prints a failure message
    followed by the list of error messages.
    """
    (ok, error_report) = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
        print('Sequence duration is: ', round(seq.duration()[0]), 's')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]


def plot_seq_kspace(seq: Sequence):
    """
    Plot the k-space trajectory of a given MRI sequence.

    Parameters
    ----------
    seq (Sequence): The MRI sequence object which contains the method `calculate_kspace`
                    that returns the k-space trajectories.
    The function calculates the k-space trajectories using the `calculate_kspace` method
    of the provided sequence object and plots the full k-space trajectory as well as
    the ADC (Analog-to-Digital Converter) sampled points.
    The plot includes:
    - Full k-space trajectory (kx vs ky)
    - ADC sampled points in k-space
    The x-axis and y-axis are labeled as 'kx [1/m]' and 'ky [1/m]' respectively.
    """
    k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

    plt.figure()
    plt.title('full k-space trajectory ($k_{x}$ x $k_{y}$)')
    plt.plot(k_traj[0], k_traj[1])
    plt.plot(k_traj_adc[0], k_traj_adc[1], '.')
    plt.xlabel('kx [1/m]')
    plt.ylabel('ky [1/m]')
    plt.show()


def plot_sequence(seq: Sequence):
    """
    Plot sequence events.

    Parameters
    ----------
    seq (Sequence): The sequence object to be plotted.
    n_enc (tuple[int, int, int]): A tuple containing three integers
    representing the encoding dimensions.

    Notes
    -----
    - If the product of the second and third elements of `n_enc` is greater than 900
    or the duration of the sequence is greater than 600,
    only the first 20% of the sequence will be plotted.
    - Otherwise, the entire sequence will be plotted.
    """
    if seq.duration()[0] > 600:
        print('Plotting only the first 20% of the sequence')
        seq.plot(time_range=(0, round(0.2 * seq.duration()[0])))
    else:
        seq.plot()


def create_filename(seq: Sequence, seq_base_name: str) -> str:
    """
    Create a filename for a given sequence with specific parameters.

    Parameters
    ----------
    seq (Sequence): The sequence object containing the definitions.
    seq_base_name (str): The base name for the sequence file.

    Returns
    -------
    str: The generated filename based on the sequence parameters.
    The filename format is:
    {seq_base_name}_{current_date}_TR{tr}_TE{te}_FOV{fov}_{n_readout}px_ETL{etl}.seq
    Where:
    - current_date: The current date in YYYYMMDD format.
    - tr: Repetition time (TR) in milliseconds.
    - te: Echo time (TE) in milliseconds.
    - fov: Field of view (FOV) in millimeters.
    - n_readout: Number of readout points.
    - etl: Echo train length.
    """
    # Add echo_time to the filename
    te = int(seq.get_definition('TE') * 1e3)
    tr = int(seq.get_definition('TR') * 1e3)
    fov = int(seq.get_definition('FOV')[0] * 1000)
    n_readout = seq.get_definition('encoding_dim')[0]
    etl = seq.get_definition('echo_train_length')
    seq_filename = seq_base_name + f'_{time.strftime("%Y%m%d")}_TR{tr}_TE{te}_FOV{fov}_{n_readout}px_ETL{etl}.seq'
    return seq_filename


def check_and_save(seq: Sequence, flags: dict, custom_name: str | None = None):
    """
    Evaluate and execute various flags for testing, plotting and writing a given sequence.

    Flags which are not part of dict are set to False by default.

    Parameters
    ----------
    seq (Sequence): The sequence object to be evaluated.
    flags (dict): A dictionary containing the following optional flags:
        - 'CHECK_TIMING' (bool): If True, check the timing of the sequence.
        - 'SHOW_TEST_REPORT' (bool): If True, print an advanced test report.
        - 'PLOT_KSPACE' (bool): If True, plot the k-space trajectory of the sequence.
        - 'PLOT_SEQUENCE' (bool): If True, plot the sequence.
        - 'WRITE_SEQ' (bool): If True, write the sequence to a file.
        - 'USE_BASE_NAME' (bool): If True, use the base name for the sequence file.
    """
    CHECK_TIMING = flags.get('CHECK_TIMING', False)
    SHOW_TEST_REPORT = flags.get('SHOW_TEST_REPORT', False)
    PLOT_KSPACE = flags.get('PLOT_KSPACE', False)
    PLOT_SEQUENCE = flags.get('PLOT_SEQUENCE', False)
    WRITE_SEQ = flags.get('WRITE_SEQ', False)
    USE_BASE_NAME = flags.get('USE_BASE_NAME', False)

    # check timing of the sequence
    if CHECK_TIMING and not SHOW_TEST_REPORT:
        check_seq_timing(seq=seq)

    # show advanced rest report
    if SHOW_TEST_REPORT:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # plot k-space trajectory
    if PLOT_KSPACE:
        plot_seq_kspace(seq=seq)

    # plot sequence
    if PLOT_SEQUENCE:
        plot_sequence(seq=seq)

    # write sequence to file
    if WRITE_SEQ:
        # define filename
        if USE_BASE_NAME:
            seq_filename = seq.get_definition('Name')
        else:
            if custom_name is not None:
                seq_filename = custom_name
            else:
                seq_filename = create_filename(seq=seq, seq_base_name=seq.get_definition('Name'))
        print('\nWriting sequence to file: ' + seq_filename)
        seq_path = Path('./applications/output_sequences/')
        seq_path.mkdir(exist_ok=True, parents=True)
        seq.write(seq_path / seq_filename)


def excitation(
    system: Opts,
    rf_type='sinc',
    fa_ex=90,
    slice_thickness=30e-3,
    rf_dur=3e-3,
    tbp=3,
    apodization=0.5,
    n_bands=3,
    band_sep=20,
):
    """Docstring."""
    # sinc pulse
    if rf_type == 'sinc':
        rf1, gz1, gzr1 = make_sinc_pulse(
            flip_angle=fa_ex * np.pi / 180,
            phase_offset=90 * np.pi / 180,
            duration=rf_dur,
            slice_thickness=slice_thickness,
            apodization=apodization,
            time_bw_product=tbp,
            system=system,
            return_gz=True,
            use='excitation',
            delay=system.rf_dead_time,
        )
    # sinc pulse
    elif rf_type == 'gauss':
        rf1, gz1, gzr1 = make_gauss_pulse(
            flip_angle=fa_ex * np.pi / 180,
            phase_offset=90 * np.pi / 180,
            duration=rf_dur,
            slice_thickness=slice_thickness,
            apodization=apodization,
            time_bw_product=tbp,
            system=system,
            return_gz=True,
            use='excitation',
            delay=system.rf_dead_time,
        )
    # SLR pulse using the sigpy.rf interface in pypulseq v1.4.0
    elif rf_type == 'slr':
        sigpy_cfg = SigpyPulseOpts(pulse_type='slr', ptype='ex')
        rf1, gz1, gzr1 = make_sigpy_pulse.sigpy_n_seq(
            flip_angle=fa_ex * np.pi / 180,
            phase_offset=90 * np.pi / 180,
            duration=rf_dur,
            slice_thickness=slice_thickness,
            time_bw_product=tbp,
            system=system,
            return_gz=True,
            pulse_cfg=sigpy_cfg,
            plot=False,
            use='excitation',
            delay=system.rf_dead_time,
        )
    # SMS pulse using the sigpy.rf interface in pypulseq v1.4.0
    elif rf_type == 'sms':
        sigpy_cfg = SigpyPulseOpts(pulse_type='sms', ptype='ex', n_bands=n_bands, band_sep=band_sep)
        rf1, gz1, gzr1 = make_sigpy_pulse.sigpy_n_seq(
            flip_angle=fa_ex * np.pi / 180,
            phase_offset=90 * np.pi / 180,
            duration=rf_dur,
            slice_thickness=slice_thickness,
            time_bw_product=tbp,
            system=system,
            return_gz=True,
            pulse_cfg=sigpy_cfg,
            plot=False,
            use='excitation',
            delay=system.rf_dead_time,
        )
        rf1.signal = (
            9.5e-3 * rf1.signal
        )  # ??? the scaling of the SMS pulse was far too high ??? reduced by trial-and-error
    else:
        raise Exception('error in excitation() - unknown pulse type: ' + rf_type)

    return rf1, gz1, gzr1
