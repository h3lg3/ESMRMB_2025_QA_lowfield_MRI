"""Run QA script on series of image data."""

from pathlib import Path

from qa.qa_report import main

# Set the path to the data directory
script_dir = Path(__file__).resolve().parent
path_data = script_dir / 'qa_data/multi_rep'
path_noise = script_dir / 'qa_data/multi_rep_noise'

main(path_data=path_data, path_noise=path_noise, phantom_diameter_m=0.11, create_report=False, transpose_dim=(2, 1, 0))
