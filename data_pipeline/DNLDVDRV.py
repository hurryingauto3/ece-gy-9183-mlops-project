#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
from datetime import timedelta

import gdown

def setup_logger():
    logger = logging.getLogger("drive_downloader")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def parse_args():
    p = argparse.ArgumentParser(description="Recursively download a Google Drive folder via gdown")
    p.add_argument("folder_id", help="Google Drive folder ID (from the URL)")
    p.add_argument("-o", "--output", default="./drive_data", help="Destination directory")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars")
    p.add_argument("--no-cookies", action="store_false", dest="use_cookies",
                   help="Disable cookies (for public folders)")
    p.add_argument("-r", "--retry", type=int, default=3, help="Max retry attempts on failure")
    return p.parse_args()

def download_folder(folder_id, output_dir, quiet, use_cookies, retries, logger):
    os.makedirs(output_dir, exist_ok=True)
    start = time.monotonic()
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{retries} → folder {folder_id}")
            gdown.download_folder(
                id=folder_id,
                output=output_dir,
                quiet=quiet,
                use_cookies=use_cookies
            )
            duration = timedelta(seconds=int(time.monotonic() - start))
            logger.info(f"Download succeeded in {duration}")
            
            # summary
            files = dirs = 0
            for _, ds, fs in os.walk(output_dir):
                dirs += len(ds)
                files += len(fs)
            logger.info(f"Saved {files} files across {dirs} subdirs in '{output_dir}'")
            return
        except Exception as e:
            logger.error(f"Error: {e}")
            if attempt < retries:
                wait = 5 * attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.critical("Max retries reached. Exiting.")
                sys.exit(1)

def main():
    logger = setup_logger()
    download_folder(
        folder_id="1Js98GAxf1LeAUTxP1JMZZIrKvyJStDgz",
        output_dir="./drive_data",
        quiet=False,
        use_cookies=True,
        retries=True,
        logger=logger
    )

if __name__ == "__main__":
    main()