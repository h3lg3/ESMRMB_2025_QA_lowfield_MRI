"""Simple QA script."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pypulseq import Sequence
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.special import erfinv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion
from skimage.morphology import disk

from utilities.sequence_helper import system


def main(
    path_data: Path,
    path_noise: str | None = None,
    path_phantom_mask: str | None = None,
    path_noise_mask: str | None = None,
    phantom_diameter_m: float = 0.12,  # bottle phantom: 0.12 m # sphere: 0.08 m
    create_report: bool = False,
    transpose_dim: tuple | None = None,
):
    """Run QA on a series of images.

    This function performs quality assurance (QA) analysis on a series of images
    from MRI data. It calculates various metrics such as SNR, uniformity, centroid
    drift, and temporal/spatial statistics. Optionally, it generates a QA report
    in PDF format. Single slice, multi slice, single repetition, and multi repetition
    supported.

    Parameters
    ----------
        path_data
            Path to the folder containing one or more folders
            with image data. Image data can be 2D, 3D or 4D,
            where the 4th dimension is the number of repetitions.
        path_noise
            Path to file with noise data.
            If None, background noise from image will be used.
        path_phantom_mask
            Path to the phantom mask file.
            If None, a mask will be generated automatically.
        path_noise_mask
            Path to the noise mask file.
            If None, a mask will be generated automatically.
        phantom_diameter_m
            Diameter of the phantom in meters.
        create_report
            Whether to create a QA report in PDF format.
        transpose_dim
            Tuple specifying the dimensions to transpose the data.

    Returns
    -------
        None
    Notes:
        - The function assumes that the readout (RO) direction is left-right and
          the phase encoding (PE) direction is up-down in the image.
        - The QA report includes metrics such as temporal SNR, spatial SNR,
          uniformity, centroid drift, and optional ghost/spike analysis.
        - The function generates visualizations and optionally saves them in the
          QA report PDF.
        - QA Options:
            GHOSTS (bool): Toggle to analyze ghost level in the PE1 direction.
            SPIKE_DETECTION (bool): Toggle to detect spikes in every slice and repetition.
            DRIFT_CORRECTION (bool): Toggle to correct for image drift in the RO direction
            over time for calculating standard deviation.
            USE_SHIFTED_MASK (bool): Toggle to shift the manual mask to the automatically
            determined image center for each time point before running analysis.
            ERODE_MASK (bool): Toggle to erode the phantom mask by 15% of the phantom
            diameter to avoid edge effects.
            MODE (int): Select which slices to analyze:
                -1: All slices
                0: Mid slice
                >0: Specific slice index
    """
    # ======
    # QA options
    # ======
    # Currently its assumed that RO is left-right and PE is up-down in image.
    GHOSTS = False  # Toggle to analyze ghost level in PE1 direction
    SPIKE_DETECTION = False  # Toggle to detect spikes in every slice and repetition
    # Toggle to correct for image drift in RO direction over time for calculating
    # standard deviation
    DRIFT_CORRECTION = True
    # Toggle to shift the manual mask to the automatically determined image center for each
    # time point before running analysis
    USE_SHIFTED_MASK = True
    ERODE_MASK = True  # Toggle to erode the phantom mask by % of the phantom diameter to avoid edge effects
    ERODE_FACTOR = 0.08  # % of phantom diameter to erode the mask
    MODE = 40  # Select which slices to analyze: -1: all slices, 0: mid slice, >0: slice index

    # ======
    # Import metadata, data and sequence
    # ======
    # Get list of all folders in data_folder
    folder_list = [f.name for f in path_data.iterdir() if f.is_dir()]
    folder_list.sort()
    n_folders = len(folder_list)
    meta_file_path = path_data / folder_list[0] / 'meta.json'
    seq_file_path = path_data / folder_list[0] / 'sequence.seq'

    # Load metadata from JSON file
    with Path.open(meta_file_path) as meta_file:
        metadata_first = json.load(meta_file)
    if n_folders > 1:  # check if more than one folder exists for import
        meta_file_path = path_data / folder_list[-1] / 'meta.json'
        with Path.open(meta_file_path) as meta_file:
            metadata_last = json.load(meta_file)
    else:
        metadata_last = metadata_first

    # Check how many repetitions/num_averages are in the data of the first folder
    num_averages = metadata_first['acquisition_parameter']['num_averages']

    # Ensure that only one folder is used for multiple repetitions or only one repetition is used for multiple folders
    if n_folders > 1:
        assert num_averages == 1, 'Only one repetition allowed for multiple folders'
    if num_averages > 1:
        assert n_folders == 1, 'Only one folder allowed for multiple repetitions'

    # Load the first image to initialize the data array
    # Matrix MxNxS, M: rows, N: columns, S: slices. 4th dimension: number of images
    # Rows is PE direction, columns is RO direction, 3rd dimension is PE2 direction, 4th dimension is number of images
    data_cmplx = np.load(path_data / folder_list[0] / 'image.npy')
    if transpose_dim is not None:
        data_cmplx = np.transpose(data_cmplx, transpose_dim)

    if num_averages > 1:
        data_cmplx = np.transpose(data_cmplx, (1, 2, 3, 0))  # transpose to [Nlin, Nadc, Npar/Nslc, num_averages]
        if MODE == 0:
            data_cmplx = data_cmplx[:, :, data_cmplx.shape[2] // 2, :]
            data_cmplx = np.expand_dims(data_cmplx, axis=-2)
        elif MODE > 0:
            data_cmplx = data_cmplx[:, :, MODE, :]
            data_cmplx = np.expand_dims(data_cmplx, axis=-2)
    else:
        if len(data_cmplx.shape) == 2:  # single slice, 2D
            data_cmplx = np.expand_dims(np.expand_dims(data_cmplx, axis=-1), axis=-1)
        elif data_cmplx.shape[2] > 1:  # multi slice, 3D
            if MODE == -1:  # analyze all slices
                data_cmplx = np.expand_dims(data_cmplx, axis=-1)
            elif MODE == 0:  # analyze mid slice
                data_cmplx = data_cmplx[:, :, data_cmplx.shape[2] // 2]
                data_cmplx = np.expand_dims(np.expand_dims(data_cmplx, axis=-1), axis=-1)
            elif MODE > 0:  # analyze specific slice
                data_cmplx = data_cmplx[:, :, MODE]
                data_cmplx = np.expand_dims(np.expand_dims(data_cmplx, axis=-1), axis=-1)

        # Import additional scans from other folders
        if n_folders > 1:
            for folder in folder_list[1:]:
                next_data = np.load(path_data / folder / 'image.npy')
                if transpose_dim is not None:
                    next_data = np.transpose(next_data, transpose_dim)
                if len(next_data.shape) == 2:
                    next_data = np.expand_dims(np.expand_dims(next_data, axis=-1), axis=-1)
                elif next_data.shape[2] > 1:  # multi slice, 3D
                    if MODE == -1:  # analyze all slices
                        next_data = np.expand_dims(next_data, axis=-1)
                    elif MODE == 0:  # analyze mid slice
                        next_data = next_data[:, :, data_cmplx.shape[2] // 2]
                        next_data = np.expand_dims(np.expand_dims(next_data, axis=-1), axis=-1)
                    elif MODE > 0:  # analyze specific slice
                        next_data = next_data[:, :, MODE]
                        next_data = np.expand_dims(np.expand_dims(next_data, axis=-1), axis=-1)
                data_cmplx = np.concatenate((data_cmplx, next_data), axis=-1)

    data = np.abs(data_cmplx)  # take absolute value of data
    mid_slice = data.shape[2] // 2
    # assert that image has equal dimensions
    assert data.shape[0] == data.shape[1], 'Image must be square'

    # load sequence file to get matrix size and resolution
    seq = Sequence(system=system)
    seq.read(seq_file_path, detect_rf_use=True)
    n_px = int(seq.get_definition('encoding_dim')[0])
    assert data.shape[0] == n_px, 'Matrix size of sequence and data must be equal'

    # ======
    # Parameters for analysis
    # ======
    n_px = data.shape[0]
    im_center = n_px // 2
    n_slices = data.shape[2]
    if len(data.shape) == 4:
        n_reps = data.shape[3]
    else:
        n_reps = 1
    resolution = seq.get_definition('FOV')[0] / n_px
    phantom_diameter_px = phantom_diameter_m / resolution
    erosion_radius = int(phantom_diameter_px * ERODE_FACTOR)  # % of phantom diameter in pixels

    # ======
    # Create masks: Phantom (regular, eroded, shifted), Noise bands, Ghost bands
    # ======
    # Either create phantom mask or load it from file
    if path_phantom_mask is None:
        # create 2d contour of size px with a circle of radius of phantom
        manual_mask = np.zeros((n_px, n_px))
        for i in range(n_px):
            for j in range(n_px):
                if (i - im_center) ** 2 + (j - im_center) ** 2 < (phantom_diameter_px / 2) ** 2:
                    manual_mask[i, j] = 1
        manual_mask = manual_mask > 0.5  # convert mask to binary
    else:
        manual_mask = np.load(path_phantom_mask)  # load contour as numpy array
        # assert that contour has same dimensions as image
        assert manual_mask.shape == data.shape, 'Contour must have same dimensions as data'

    # Erode the mask
    if ERODE_MASK:
        eroded_mask = binary_erosion(manual_mask, disk(erosion_radius))
    else:
        eroded_mask = binary_erosion(manual_mask, disk(0))

    # Define width noise bands
    _width_noise_band = int(np.floor((n_px - phantom_diameter_px) / 4))

    # Create Ghost Mask
    if GHOSTS:
        ghost_mask = np.zeros((n_px, n_px))
        ghost_mask[:_width_noise_band, :] = 1  # ghost band should be along PE direction
        ghost_mask[-_width_noise_band:, :] = 1
        ghost_mask = ghost_mask > 0.5

    # Either use noise scan, create noise mask or load noise mask from file
    if path_noise is not None:
        data_noise = np.load(path_noise / 'image.npy')
        data_noise = np.abs(data_noise)  # take absolute value of noise
        if transpose_dim is not None:
            data_noise = np.transpose(data_noise, transpose_dim)  # transpose for consistency with data / plotting
        noise_info = 'Noise scan provided.'
    else:
        if path_noise_mask is None:
            noise_mask = np.zeros((n_px, n_px))
            noise_mask[:, :_width_noise_band] = 1  # noise band should be along readout direction
            noise_mask[:, -_width_noise_band:] = 1
            noise_mask = noise_mask > 0.5
            noise_info = 'Noise based on background signal.'
        else:
            noise_mask = np.load(path_noise_mask)  # load noise mask as numpy array
            # assert that noise mask has same dimensions as image
            assert noise_mask.shape == data.shape, 'Noise mask must have same dimensions as image'
            noise_info = 'Noise based on noise mask.'

    # ======
    # Image center shift over time
    # ======
    # Calculate centroid shift for each image
    # smooth data_cmplx slice wise
    data_smooth = np.zeros(data_cmplx.shape, dtype=np.complex64)
    for i in range(n_slices):
        for j in range(n_reps):
            _temp = data_cmplx[:, :, i, j]
            data_smooth[:, :, i, j] = gaussian_filter(_temp, sigma=1)
    data_smooth = np.abs(data_cmplx)  # take absolute value of data

    X, Y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    centroid_shift = np.zeros((n_slices, n_reps))
    center_RO = np.zeros((n_slices, n_reps))
    center_PE = np.zeros((n_slices, n_reps))
    for i in range(n_slices):
        for j in range(n_reps):
            # find center of gravity of image using smoothed data
            _temp = np.squeeze(data_smooth[:, :, i, j])
            optimal_threshold = threshold_otsu(_temp) * 0.8  # lower threshold
            auto_mask = _temp > optimal_threshold
            auto_mask = binary_fill_holes(auto_mask)

            # calculate l2 norm between (x_center, y_center) and center of auto mask
            RO_center = np.sum(X * auto_mask) / np.sum(auto_mask)
            PE_center = np.sum(Y * auto_mask) / np.sum(auto_mask)
            center_RO[i, j] = RO_center
            center_PE[i, j] = PE_center
            centroid_shift[i, j] = np.sqrt((RO_center - im_center) ** 2 + (PE_center - im_center) ** 2)

    # Shift manual and eroded mask to center of gravity, only makes sense for single slice
    if MODE >= 0 and USE_SHIFTED_MASK:
        manual_mask_shifted = np.zeros((n_px, n_px, n_reps))
        eroded_mask_shifted = np.zeros((n_px, n_px, n_reps))
        for j in range(n_reps):
            manual_mask_shifted[:, :, j] = np.roll(
                manual_mask, (int(center_PE[0, j] - im_center), int(center_RO[0, j] - im_center)), axis=(0, 1)
            )
            eroded_mask_shifted[:, :, j] = np.roll(
                eroded_mask, (int(center_PE[0, j] - im_center), int(center_RO[0, j] - im_center)), axis=(0, 1)
            )
        manual_mask_shifted = manual_mask_shifted > 0.5  # convert mask to binary
        eroded_mask_shifted = eroded_mask_shifted > 0.5

    # ======
    # Image center drift in RO direction [px/TR] for each slice
    # ======
    centroid_drift = []  # for each slice
    if n_reps > 1:
        for i in range(n_slices):
            # Linear fit (degree 1)
            _coefficients = np.polyfit(np.arange(n_reps), np.array(center_RO[i, :]), 1)
            centroid_drift.append(_coefficients[0])

    # ======
    # Spatial SNR for each image: time points and slices
    # ======
    # calculate the mean and standard deviation of the background
    snr = np.zeros((n_slices, n_reps))
    phantom_mean = np.zeros((n_slices, n_reps))
    phantom_std = np.zeros((n_slices, n_reps))
    std_noise = np.zeros((n_slices, n_reps))
    mean_noise = np.zeros((n_slices, n_reps))
    for i in range(n_slices):
        for j in range(n_reps):
            if MODE >= 0 and USE_SHIFTED_MASK:
                _mask = eroded_mask_shifted[:, :, j]
            else:
                _mask = eroded_mask
            _temp = data[:, :, i, j]
            if path_noise is not None:
                std_noise[i, j] = np.std(data_noise[:])
                mean_noise[i, j] = np.mean(data_noise[:])
            else:
                _noise_signal = _temp[noise_mask]
                std_noise[i, j] = np.std(_noise_signal)
                mean_noise[i, j] = np.mean(_noise_signal)
            _mean_signal = np.mean(_temp[_mask])
            phantom_mean[i, j] = _mean_signal  # calculate the mean signal in phantom for each slice and repetition
            phantom_std[i, j] = np.std(
                _temp[_mask]
            )  # calculate the std signal in phantom for each slice and repetition
            snr[i, j] = np.divide(_mean_signal, std_noise[i, j])  # calculate the SNR

    # ======
    # Temporal Standard Deviation and SNR for each slice
    # ======
    image_mean = np.mean(data, axis=3)
    image_std = np.std(data, axis=3)
    if DRIFT_CORRECTION:
        # Drift correction of time course within central phantom ROI for calculating standard deviation
        _data_drift_corr = data.copy()
        b, a = butter(3, 0.2, btype='high')
        for y in range(int(im_center - phantom_diameter_px // 2 - 1), int(im_center + phantom_diameter_px // 2)):
            for x in range(
                int(im_center - phantom_diameter_px // 2 - 1), int(im_center // 2 + phantom_diameter_px // 2)
            ):
                for z in range(n_slices):
                    _data_drift_corr[y, x, z, :] = filtfilt(b, a, data[y, x, z, :], padlen=n_reps - 1)

        image_std_corr = np.std(_data_drift_corr, axis=3)
    else:
        image_std_corr = image_std  # no drift correction

    tempororal_snr = []
    mean_temporal_snr = []
    tmean_phantom = []
    tstd_phantom = []
    for i in range(n_slices):
        _temp = np.divide(image_mean[:, :, i], image_std_corr[:, :, i])
        tempororal_snr.append(_temp)
        mean_temporal_snr.append(np.mean(_temp[eroded_mask]))  # temporal snr

        _temp = image_std_corr[:, :, i]
        tstd_phantom.append(np.mean(_temp[eroded_mask]))  # voxelwise std over time

        _temp = image_mean[:, :, i]
        tmean_phantom.append(np.mean(_temp[eroded_mask]))  # voxelwise mean over time

    # ======
    # Uniformity for each image: time points and slices
    # ======
    uniformity = np.zeros((n_slices, n_reps))  # for each slice and repetition
    mean_uniformity = []  # for each slice, averaged over repetitions
    for i in range(n_slices):
        for j in range(n_reps):
            if MODE >= 0 and USE_SHIFTED_MASK:
                _mask = eroded_mask_shifted[:, :, j]
            else:
                _mask = eroded_mask
            _temp = data[:, :, i, j]
            # calculate the uniformity of the signal inside the phantom mask
            uniformity[i, j] = 100 * (
                1 - (np.max(_temp[_mask]) - np.min(_temp[_mask])) / (np.max(_temp[_mask]) + np.min(_temp[_mask]))
            )
        mean_uniformity.append(np.mean(uniformity[i, :], axis=-1))

    # ======
    # Ghost level for each time point
    # ======
    if GHOSTS:
        ghost_signal = np.zeros((n_slices, n_reps))
        ghost_to_image_ratio = np.zeros((n_slices, n_reps))
        mean_ghost = []
        for i in range(n_slices):
            for j in range(n_reps):
                _temp = data[:, :, i, j]
                ghost_signal[i, j] = np.mean(_temp[ghost_mask])  # mean signal in ghost bands
                ghost_to_image_ratio[i, j] = (
                    np.mean(_temp[ghost_mask])
                    / phantom_mean[i, j]
                    * 100  # ratio of signal in ghost bands to signal in phantom
                )  # use abs value to compare to image signal
            mean_ghost.append(np.mean(ghost_signal[i, :]))  # mean ghost signal over all repetitions for each slice

    # ======
    # Spike level for each repetition
    # ======
    if SPIKE_DETECTION:
        sens = -erfinv(1 / (n_slices * n_reps) - 1) * np.sqrt(2) + 1.0  # threshold in units of std (+ offset [1.0])
        # Scaled with number of samples (reps*slices) to prevent false alarms = (1 - erf(sens/sqrt(2)))*slices*rep
        # http://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data

        # Mean of whole image
        full_image_mean = np.zeros((n_slices, n_reps))
        for i in range(n_slices):
            for j in range(n_reps):
                _temp = data[:, :, i, j]
                full_image_mean[i, j] = np.mean(_temp[:])
        # DETREND timecourse
        # Precisely zero-phase distortion with the "filtfilt" method
        # (spike position not shifted! *but* filter effects in time steps around spike!!)
        b, a = butter(3, 0.2, btype='high')
        for i in range(n_slices):
            full_image_mean[i, :] = filtfilt(b, a, full_image_mean[i, :], padlen=n_reps - 1)

        # Calculate mean and standard deviation
        M2 = np.mean(full_image_mean, axis=1)
        M3 = np.std(full_image_mean, axis=1)
        M4 = np.zeros((n_slices, n_reps))
        n_spikes = 0
        for i in range(n_slices):
            for j in range(n_reps):
                if abs(full_image_mean[i, j] - M2[i]) > sens * M3[i]:
                    M4[i, j] = full_image_mean[i, j]
                    n_spikes += 1
                    plt.figure(22)
                    plt.subplot(221)
                    plt.imshow(np.log(data[:, :, i, j]), vmin=2, vmax=8, cmap='gray')
                    plt.axis('off')
                    plt.title(f'detected spikes in slc {j} rep {i} (thres={sens} sigma)')
                    plt.subplot(222)
                    plt.plot(full_image_mean)
                    plt.plot(i, full_image_mean[i], 'r*')
                    plt.title(f'timecourse (spike# {n_spikes})')
                    plt.plot(full_image_mean[i, :])
                    plt.plot(j, full_image_mean[i, j], 'r*')
                    plt.subplot(223)
                    plt.imshow(M4)
                    plt.ylabel('slice #')
                    plt.xlabel('repetition')
                    plt.pause(0.2)

    # ======
    # Results for QA report
    # ======
    # Summary of QA analysis
    results = [
        ['Metric', 'Value', 'Description'],
        [
            'Timestamp',
            f'{metadata_first["date_time"]}\n{metadata_last["date_time"]}',
            'Date and time of first and last scan',
        ],
        [
            'Acquisition info',
            f'{max(len(folder_list), num_averages)},\n{list(
                seq.get_definition('FOV'))}\n{list(
                seq.get_definition('encoding_dim'))}',
            'Number of scans found\nField of view in mm\nMatrix size',
        ],
        [
            'Noise info',
            f'mean = {np.mean(mean_noise[mid_slice, :]):.1f}, std = {np.mean(std_noise[mid_slice, :]):.1f}',
            noise_info,
        ],
        [
            'Temporal SNR',
            f'{mean_temporal_snr[mid_slice]:.2f}',
            'Voxel-wise mean and std over time, mid slice, spatially averaged',
        ],
        [
            'Spatial SNR',
            f'{np.mean(snr[mid_slice, :]):.2f}',
            'Mean phatom (spatial) and std in noise bands, mid slice, repetitions averaged',
        ],
        [
            'Uniformity',
            f'{mean_uniformity[mid_slice]:.2f} %',
            'Signal variation inside phantom, mid slice, repetitions averaged',
        ],
    ]

    if GHOSTS:
        results.append(['Ghost Signal', f'{mean_ghost[mid_slice]:.2f}, mid slice, all reps', ''])

    if n_reps > 1:
        results.append(
            [
                'Centroid Drift in RO',
                f'{centroid_drift[mid_slice]:.2f} px/rep',
                'Drift of image center in RO direction over time, mid slice',
            ]
        )

    if SPIKE_DETECTION:
        results.append(['Spikes Detected', f'{n_spikes}', 'Spike detection for each slice and repetition separately'])

    results.append(['QA Settings', 'GHOSTS, SPIKE_DETECTION', f'{GHOSTS}, {SPIKE_DETECTION}'])
    results.append(
        [
            'QA Adjustments',
            'DRIFT_CORRECTION\nUSE_SHIFTED_MASK, ERODE_MASK',
            f'{DRIFT_CORRECTION}\n{USE_SHIFTED_MASK}, {ERODE_MASK}',
        ]
    )
    if MODE == -1:
        results.append(['Slices', 'MODE', 'All slices'])
    elif MODE == 0:
        results.append(['Slices', 'MODE', 'Mid slice'])
    elif MODE > 0:
        results.append(['Slices', 'MODE', f'Slice {MODE}'])

    # Prepare table to append metadata to PDF
    meta_data_list = []
    for _, (key, value) in enumerate(metadata_first.items()):
        if key in ['acquisition_parameter', 'sequence', 'info', 'dwell_time']:
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if key2 == 'sequence_parameter':
                        for key3, value3 in value2.items():
                            meta_data_list.append([key3, value3])
                    else:
                        meta_data_list.append([key2, value2])
            else:
                meta_data_list.append([key, value])

    # Write results to an Excel sheet
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results[1:], columns=results[0])

    # Save the results to an Excel file
    excel_path = Path(path_data / ('QA_results_' + folder_list[0] + '.xlsx'))
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name='QA Summary', index=False)

        # Add metadata to a separate sheet
        metadata_df = pd.DataFrame(meta_data_list, columns=['Parameter', 'Value'])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

    # Create a PDF document
    fontsize = 9
    plt.rcParams.update({'font.size': fontsize})  # Update default font size
    plt.rcParams['axes.titlesize'] = fontsize  # Update title font size
    plt.rcParams['axes.labelsize'] = fontsize  # Update label font size
    plt.rcParams['xtick.labelsize'] = fontsize  # Update x tick font size
    plt.rcParams['ytick.labelsize'] = fontsize  # Update y tick font size
    plt.rcParams['legend.fontsize'] = fontsize  # Update legend font size
    plt.rcParams['figure.titlesize'] = fontsize  # Update figure title font size

    with PdfPages(Path(path_data / ('QA_report_' + folder_list[0] + '.pdf'))) as pdf:
        # Add results table to the first page of the PDF
        if create_report:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis('off')
            table_data = [list(row) for row in results]
            table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='left', edges='horizontal')
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.auto_set_column_width(col=list(range(len(results[0]))))
            table.scale(1, 3)
            ax.set_title('QA Summary', fontsize=fontsize, pad=20)
            pdf.savefig(fig)
            plt.close(fig)

        # ======
        # Visualization: Images for QA
        # ======
        fig = plt.figure(figsize=(10.4, 7))
        if n_reps > 1:
            fig.suptitle('Visualization')
        else:
            fig.suptitle('Visualization: Single repetition')

        plt.subplot(2, 2, 1)
        plt.imshow(image_mean[:, :, mid_slice], cmap='gray')
        plt.colorbar()
        plt.ylabel('PE1 direction')
        plt.xlabel('RO direction')
        # plt.title(
        #     f'Image averaged over repetitions\nMean Signal: {tmean_phantom[mid_slice]:.1f}, \
        #     Uniformity: {mean_uniformity[mid_slice]:.1f}%'
        # )
        plt.clim(vmin=0, vmax=100)
        plt.subplot(2, 2, 2)
        plt.imshow(image_std[:, :, mid_slice], cmap='gray')
        plt.colorbar()
        plt.clim(vmin=0, vmax=20)
        plt.ylabel('PE1 direction')
        plt.xlabel('RO direction')
        # plt.title(f'std over repetitions\nMean std: {tstd_phantom[mid_slice]:.1f}')

        plt.subplot(2, 2, 3)
        plt.imshow(data[:, :, mid_slice, 0], cmap='gray')
        plt.clim(vmin=0, vmax=100)
        plt.colorbar()
        plt.contour(manual_mask, colors='red', levels=[0.5])
        if MODE >= 0 and USE_SHIFTED_MASK:
            plt.contour(manual_mask_shifted[:, :, mid_slice], colors='blue', levels=[0.5])
            plt.contour(eroded_mask_shifted[:, :, mid_slice], colors='blue', levels=[0.5], linestyles='dotted')
        if path_noise is None:  # plot noise mask only if not using noise scan
            plt.contour(noise_mask, colors='purple', levels=[0.5], linestyles='dotted')
        if GHOSTS:
            plt.contour(ghost_mask, colors='green', levels=[0.5], linestyles='dotted')
        plt.scatter([im_center], [im_center], color='red', marker='x')
        plt.scatter([center_RO[mid_slice, 0]], [center_PE[mid_slice, 0]], color='blue', marker='x')
        # plt.title('Masks for phantom/noise and centroid')
        plt.ylabel('PE1 direction')
        plt.xlabel('RO direction')

        if n_reps > 1:
            plt.subplot(2, 2, 4)
            _coefficients = np.polyfit(np.arange(n_reps), np.array(center_RO[mid_slice, :]), 1)
            plt.plot(np.arange(n_reps), center_RO[mid_slice, :], 'o')
            plt.plot(np.arange(n_reps), np.polyval(_coefficients, np.arange(n_reps)), 'r-')
            # plt.title('Centroid drift in RO in middle slice')
            plt.xlabel('Repetition')
            plt.ylabel('Centroid position [px]')
            plt.grid(axis='y')
            # plt.ylim((np.floor(np.min(center_RO[mid_slice, :])), np.ceil(np.max(center_RO[mid_slice, :]))))

        if create_report:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

        # ======
        # Save QA Metrics for middle slice over all repetitions
        # ======
        if GHOSTS:
            fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(11.7, 8.3))
        if n_reps > 1:
            fig.suptitle('QA for middle slice and all repetitions')
        else:
            fig.suptitle('Single repetition QA, middle slice')

        axs[0, 0].plot(range(n_reps), snr[mid_slice, :], marker='o')
        # axs[0, 0].set_title('Phantom signal to background noise (RO dir.)')
        axs[0, 0].set_ylabel('SNR')
        axs[0, 0].set_xlabel('Repetition')
        axs[0, 0].grid(axis='y')

        ax2 = axs[0, 1].twinx()
        axs[0, 1].plot(range(n_reps), phantom_mean[mid_slice, :], marker='o')
        ax2.plot(range(n_reps), phantom_std[mid_slice, :], marker='x', color='red')
        # axs[0, 1].set_title('Mean phantom signal (blue) and std (red)')
        axs[0, 1].set_ylabel('Mean signal')
        axs[0, 1].set_xlabel('Repetition')
        axs[0, 1].grid(axis='y')
        ax2.tick_params(axis='y', labelcolor='red')

        axs[1, 0].plot(range(n_reps), centroid_shift[mid_slice, :], marker='o')
        # axs[1, 0].set_title('Centroid offset (RO+PE1) from center')
        axs[1, 0].set_xlabel('Repetition')
        axs[1, 0].set_ylabel('Distance to center [px]')
        axs[1, 0].grid(axis='y')

        axs[1, 1].plot(range(n_reps), uniformity[mid_slice, :], marker='o')
        # axs[1, 1].set_title('Signal variation (range) in phantom')
        axs[1, 1].set_xlabel('Repetition')
        axs[1, 1].set_ylabel('Signal uniformity [%]')
        axs[1, 1].grid(axis='y')

        if GHOSTS:
            axs[0, 2].plot(range(n_reps), ghost_signal[mid_slice, :], marker='o')
            axs[0, 2].set_title('Mean signal in ghost bands (PE1 dir.)')
            axs[0, 2].yaxis.tick_right()
            axs[0, 2].yaxis.set_label_position('right')
            axs[0, 2].set_ylabel('Ghost Signal')

            axs[1, 2].plot(range(n_reps), ghost_to_image_ratio[mid_slice, :], marker='o')
            axs[1, 2].set_title('Mean ghost signal to mean image signal')
            axs[1, 2].set_xlabel('Repetition')
            axs[1, 2].set_ylabel('Ratio (%)')

        if create_report:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

        # ======
        # Visualization: Phantom in 3 planes
        # ======
        # if create_report:
        fig = plt.figure(figsize=(11.7, 8.3))
        if n_slices > 1:
            fig.suptitle('Visualization: 3 Planes of the phantom')
        else:
            fig.suptitle('Visualization: Single slice (2D) analysis')
        plt.subplot(2, 2, 1)
        plt.imshow(image_mean[:, :, mid_slice], cmap='gray')
        plt.clim(vmin=np.min(image_mean[:, :, mid_slice]), vmax=np.max(image_mean[:, :, mid_slice]))
        plt.colorbar()
        plt.xlabel('RO direction')
        plt.ylabel('PE1 direction')
        plt.title('RO-PE1 Plane')
        plt.subplot(2, 2, 2)
        plt.imshow(image_mean[im_center, :, :], cmap='gray')
        plt.clim(vmin=np.min(image_mean[:, :, mid_slice]), vmax=np.max(image_mean[:, :, mid_slice]))
        plt.colorbar()
        plt.title('RO-PE2 Plane')
        plt.xlabel('PE2 direction')
        plt.ylabel('RO direction')
        plt.subplot(2, 2, 3)
        plt.imshow(image_mean[:, im_center, :], cmap='gray')
        plt.clim(vmin=np.min(image_mean[:, :, mid_slice]), vmax=np.max(image_mean[:, :, mid_slice]))
        plt.colorbar()
        plt.title('PE1-PE2 Plane')
        plt.xlabel('PE2 direction')
        plt.ylabel('PE1 direction')
        if path_noise is not None:
            plt.subplot(2, 2, 4)
            plt.imshow(data_noise[:, :, 0], cmap='gray')
            plt.clim(vmin=0, vmax=np.ceil(np.max(data_noise[:, :, 0])))
            plt.colorbar()
            # plt.title(
            #     f'Noise Scan, mean = {mean_noise[0,0]:.1f}, std = {std_noise[0,0]:.1f}'
            # )  # all entries are the same
            plt.xlabel('RO direction')
            plt.ylabel('PE1 direction')
        if create_report:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

        if create_report:
            # Add metadata to report
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis('off')
            table_data = [list(row) for row in meta_data_list]
            table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='left', edges='horizontal')
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.auto_set_column_width(col=list(range(len(results[0]))))
            table.scale(1, 1.2)
            ax.set_title('Metadata', fontsize=fontsize, pad=20)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    main(
        path_data='../OSI_data/OSI_scans/220425_Flaschenphantom/4mm/',
    )

