# -*- coding: utf-8 -*-
# @Date    : 2024-10-20
# @Author  : MoshiQAQ & didi
# @Desc    : Download and extract dataset files

import os
import sys
import tarfile
import shutil
from typing import Dict

import requests
from tqdm import tqdm

# Add the parent directory to sys.path to import scripts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logs import logger


def download_file(url: str, filename: str) -> None:
    """Download a file from the given URL and show progress."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(filename, "wb") as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()


def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def process_dataset(url: str, filename: str, extract_path: str, force: bool = False) -> None:
    """Download, extract, and clean up a dataset if needed."""
    if os.path.exists(extract_path):
        if force:
            logger.info(f"Force mode: Removing existing {extract_path}...")
            shutil.rmtree(extract_path)
        else:
            logger.info(f"{extract_path} already exists. Skipping download and extraction.")
            return

    download_needed = not os.path.exists(filename)

    if download_needed:
        logger.info(f"Downloading {filename}...")
        download_file(url, filename)
    else:
        logger.info(f"Using existing {filename}...")

    logger.info(f"Extracting {filename} to {extract_path}...")
    extract_tar_gz(filename, extract_path)

    if os.path.exists(filename):
        os.remove(filename)
        logger.info(f"Removed {filename}")
    else:
        logger.warning(f"{filename} not found during cleanup.")

    logger.info(f"Processing {filename} completed.")


# Define the datasets to be downloaded
datasets_to_download: Dict[str, Dict[str, str]] = {
    "datasets": {
        "url": "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e",
        "filename": "aflow_data.tar.gz",
        "extract_path": "data/datasets",
    },
    "results": {
        "url": "https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv",
        "filename": "result.tar.gz",
        "extract_path": "data/results",
    },
    "initial_rounds": {
        "url": "https://drive.google.com/uc?export=download&id=1UBoW4WBWjX2gs4I_jq3ALdXeLdwDJMdP",
        "filename": "initial_rounds.tar.gz",
        "extract_path": "workspace",
    },
}


def download(required_datasets, force_download: bool = False):
    """Main function to process all selected datasets"""
    for dataset_name in required_datasets:
        dataset = datasets_to_download[dataset_name]
        url = dataset["url"]
        filename = dataset["filename"]
        extract_path = dataset["extract_path"]

        process_needed = force_download or not os.path.exists(extract_path)
        if process_needed:
            logger.info(f"Processing dataset: {dataset_name}")
            process_dataset(url, filename, extract_path, force=force_download)
        else:
            logger.info(f"Skipping {dataset_name} as {extract_path} already exists.")


def test_download():
    """Test function to verify download and extraction process"""
    test_datasets = ["datasets", "results", "initial_rounds"]
    for name in test_datasets:
        dataset = datasets_to_download[name]
        extract_path = dataset["extract_path"]
        filename = dataset["filename"]

        # Cleanup before test
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        if os.path.exists(filename):
            os.remove(filename)

    # Test normal download
    download(test_datasets, force_download=False)
    for name in test_datasets:
        dataset = datasets_to_download[name]
        assert os.path.exists(dataset["extract_path"]), f"{dataset['extract_path']} not created."

    # Test skipping existing
    download(test_datasets, force_download=False)

    # Test force download
    download(test_datasets, force_download=True)
    for name in test_datasets:
        dataset = datasets_to_download[name]
        assert os.path.exists(dataset["extract_path"]), f"{dataset['extract_path']} not recreated."

    # Cleanup after test
    for name in test_datasets:
        dataset = datasets_to_download[name]
        if os.path.exists(dataset["extract_path"]):
            shutil.rmtree(dataset["extract_path"])
        if os.path.exists(dataset["filename"]):
            os.remove(dataset["filename"])

    logger.info("All download tests passed.")


if __name__ == "__main__":
    test_download()