# pyre-strict

"""
Generic Kaggle Dataset Extractor (No API Required)

This module provides utilities to download and extract data from ANY Kaggle
competition or dataset without requiring the Kaggle API.

MANUAL DOWNLOAD INSTRUCTIONS:
-----------------------------
1. Go to the Kaggle competition/dataset page
2. Sign in to your Kaggle account (create one if needed)
3. Accept the competition rules if prompted
4. Click "Download All" or download individual files
5. Extract the downloaded zip file(s) to your data directory
6. Use this module to load and process the data
"""

import json
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd


logger: logging.Logger = logging.getLogger(__name__)


class KaggleDataset:
    """
    Generic handler for loading and processing any Kaggle dataset.

    Works with both competitions and datasets.

    Attributes:
        data_dir: Directory where the dataset is stored
        dataset_name: Name of the dataset/competition
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: Optional[str] = None,
        kaggle_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the dataset handler.

        Args:
            dataset_name: Name of the Kaggle competition or dataset
                         (e.g., 'iwildcam2022-fgvc9', 'titanic', 'house-prices-advanced-regression-techniques')
            data_dir: Directory path where the dataset is stored.
                     If None, uses './{dataset_name}_data'
            kaggle_url: Full URL to the Kaggle dataset page (optional)

        Examples:
            >>> # For a competition
            >>> dataset = KaggleDataset("titanic")

            >>> # For a dataset
            >>> dataset = KaggleDataset("netflix-shows", kaggle_url="https://www.kaggle.com/datasets/shivamb/netflix-shows")

            >>> # With custom data directory
            >>> dataset = KaggleDataset("titanic", data_dir="./my_data")
        """
        self.dataset_name: str = dataset_name

        if data_dir is None:
            data_dir = f"./{dataset_name}_data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect URL if not provided
        if kaggle_url:
            self.kaggle_url: str = kaggle_url
        else:
            # Assume it's a competition by default
            self.kaggle_url: str = (
                f"https://www.kaggle.com/competitions/{dataset_name}/data"
            )

    def download_with_curl(self, cookies_file: str) -> None:
        """
        Download dataset using curl with Kaggle cookies.

        This method requires you to export your browser cookies after logging
        into Kaggle. Use a browser extension like "Get cookies.txt" to export
        cookies to a file.

        Args:
            cookies_file: Path to cookies.txt file exported from browser
        """
        # Determine download URL based on whether it's a competition or dataset
        if "/competitions/" in self.kaggle_url:
            download_url = (
                f"https://www.kaggle.com/competitions/{self.dataset_name}/data/download"
            )
        else:
            # For datasets, extract the dataset path
            download_url = self.kaggle_url.replace("/datasets/", "/datasets/download/")

        output_file = self.data_dir / f"{self.dataset_name}.zip"

        cmd = [
            "curl",
            "-L",
            "-o",
            str(output_file),
            "-b",
            cookies_file,
            download_url,
        ]

        logger.info(f"Downloading dataset to {output_file}")
        subprocess.run(cmd, check=True)

        if output_file.exists():
            self._extract_zip(output_file)

    def _extract_zip(self, zip_path: Path, remove_after: bool = False) -> None:
        """Extract a zip file to the data directory."""
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        if remove_after:
            zip_path.unlink()

        logger.info("Extraction complete!")

    def extract_downloaded_zip(
        self,
        zip_path: Optional[str] = None,
        remove_after: bool = False,
    ) -> None:
        """
        Extract a manually downloaded zip file.

        Args:
            zip_path: Path to the downloaded zip file. If None, searches for
                     common zip file names in data_dir and current directory.
            remove_after: If True, delete the zip file after extraction.
        """
        if zip_path is None:
            # Look for common zip file names
            possible_paths = [
                self.data_dir / f"{self.dataset_name}.zip",
                Path(f"{self.dataset_name}.zip"),
                self.data_dir / "archive.zip",
                Path("archive.zip"),
                Path.home() / "Downloads" / f"{self.dataset_name}.zip",
                Path.home() / "Downloads" / "archive.zip",
            ]
            for p in possible_paths:
                if p.exists():
                    zip_path = str(p)
                    break

        if zip_path is None or not Path(zip_path).exists():
            raise FileNotFoundError(
                f"No zip file found. Please download from {self.kaggle_url} "
                "and provide the path to the zip file."
            )

        self._extract_zip(Path(zip_path), remove_after=remove_after)

    def list_files(self) -> list[str]:
        """
        List all files in the data directory.

        Returns:
            List of file paths relative to data_dir
        """
        files = []
        for f in self.data_dir.rglob("*"):
            if f.is_file():
                files.append(str(f.relative_to(self.data_dir)))
        return sorted(files)

    def list_csv_files(self) -> list[str]:
        """List all CSV files in the data directory."""
        return [f for f in self.list_files() if f.endswith(".csv")]

    def list_json_files(self) -> list[str]:
        """List all JSON files in the data directory."""
        return [f for f in self.list_files() if f.endswith(".json")]

    def print_download_instructions(self) -> None:
        """Print instructions for manually downloading the dataset."""
        print(
            f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Kaggle Dataset Download Instructions                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Dataset: {self.dataset_name:<55}      ║
║                                                                              ║
║  1. Open your browser and go to:                                             ║
║     {self.kaggle_url:<66} ║
║                                                                              ║
║  2. Sign in to your Kaggle account (or create one)                           ║
║                                                                              ║
║  3. Accept the competition rules if prompted                                 ║
║                                                                              ║
║  4. Click "Download All" or select individual files                          ║
║                                                                              ║
║  5. Move the downloaded file(s) to:                                          ║
║     {str(self.data_dir):<66} ║
║                                                                              ║
║  6. If you downloaded a zip file, extract it with:                           ║
║     dataset.extract_downloaded_zip("path/to/downloaded.zip")                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        )

    def load_csv(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load a CSV file from the data directory.

        Args:
            filename: Name of the CSV file (e.g., 'train.csv', 'test.csv')
            **kwargs: Additional arguments to pass to pd.read_csv()

        Returns:
            DataFrame containing the CSV data

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            # Try to find in subdirectories
            matches = list(self.data_dir.rglob(filename))
            if matches:
                file_path = matches[0]
            else:
                available = self.list_csv_files()
                self.print_download_instructions()
                raise FileNotFoundError(
                    f"'{filename}' not found in {self.data_dir}. "
                    f"Available CSV files: {available}"
                )

        logger.info(f"Loading {file_path}")
        return pd.read_csv(file_path, **kwargs)

    def load_json(self, filename: str) -> dict[str, Any]:
        """
        Load a JSON file from the data directory.

        Args:
            filename: Name of the JSON file

        Returns:
            Dictionary containing the JSON data
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            # Try to find in subdirectories
            matches = list(self.data_dir.rglob(filename))
            if matches:
                file_path = matches[0]
            else:
                available = self.list_json_files()
                self.print_download_instructions()
                raise FileNotFoundError(
                    f"'{filename}' not found in {self.data_dir}. "
                    f"Available JSON files: {available}"
                )

        logger.info(f"Loading {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    def load_all_csvs(self) -> dict[str, pd.DataFrame]:
        """
        Load all CSV files in the data directory.

        Returns:
            Dictionary mapping filename to DataFrame
        """
        result = {}
        for csv_file in self.list_csv_files():
            name = Path(csv_file).stem  # Get filename without extension
            result[name] = self.load_csv(csv_file)
        return result

    def get_dataset_info(self) -> dict[str, Any]:
        """
        Get summary information about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        all_files = self.list_files()
        csv_files = self.list_csv_files()
        json_files = self.list_json_files()

        info: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "kaggle_url": self.kaggle_url,
            "data_dir": str(self.data_dir),
            "total_files": len(all_files),
            "csv_files": csv_files,
            "json_files": json_files,
            "other_files": [
                f for f in all_files if f not in csv_files and f not in json_files
            ],
        }

        # Add file sizes
        total_size = 0
        for f in all_files:
            file_path = self.data_dir / f
            if file_path.exists():
                size = file_path.stat().st_size
                total_size += size

        info["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return info

    def preview_csv(self, filename: str, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview first n rows of a CSV file.

        Args:
            filename: Name of the CSV file
            n_rows: Number of rows to preview

        Returns:
            DataFrame with first n rows
        """
        return self.load_csv(filename, nrows=n_rows)

    def get_csv_info(self, filename: str) -> dict[str, Any]:
        """
        Get information about a CSV file without loading it fully.

        Args:
            filename: Name of the CSV file

        Returns:
            Dictionary with columns, dtypes, and row count
        """
        df = self.load_csv(filename)
        return {
            "filename": filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }


# ============================================================================
# Convenience functions for common datasets
# ============================================================================


def load_kaggle_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None,
    kaggle_url: Optional[str] = None,
) -> KaggleDataset:
    """
    Quick function to load any Kaggle dataset.

    Args:
        dataset_name: Name of the competition or dataset
        data_dir: Where to store/load data
        kaggle_url: Full Kaggle URL (optional)

    Returns:
        KaggleDataset instance

    Examples:
        >>> # Titanic competition
        >>> ds = load_kaggle_dataset("titanic")
        >>> train = ds.load_csv("train.csv")

        >>> # House prices competition
        >>> ds = load_kaggle_dataset("house-prices-advanced-regression-techniques")
        >>> train = ds.load_csv("train.csv")

        >>> # Custom dataset
        >>> ds = load_kaggle_dataset(
        ...     "netflix-shows",
        ...     kaggle_url="https://www.kaggle.com/datasets/shivamb/netflix-shows"
        ... )
    """
    return KaggleDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        kaggle_url=kaggle_url,
    )
