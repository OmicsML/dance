import os
import urllib.request
import zipfile

import tqdm

from dance import logger


def delete_file(path):
    """Delete a file.

    Parameters
    ----------
    path
        Path to which the file will to removed.

    """
    if not os.path.exists(path):
        logger.info("File does not exist")
    else:
        logger.info(f"Deleting {path}")
        os.remove(path)


def download_file(url, path):
    """Download a file given the url to the specified path.

    Parameters
    ----------
    url
        URL from which the data will be downloaded.
    path
        Path to which the downloaded data will be written.

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        u = urllib.request.urlopen(url)
        f = open(path, "wb")
        meta = u.info()
        file_size = int(meta.get("Content-Length", 0))
        logger.info(f"Downloading: {path} Bytes: {file_size:,}")

        file_size_dl = 0
        block_sz = 8192
        with tqdm.tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                bar.update(len(buffer))
        f.close()
        u.close()
        return True
    else:
        logger.info("File already downloaded")
        return False


def unzip_file(path, directory_to_extract_to):
    """Extract content in a zipped file.

    Parameters
    ----------
    path
        Path to the zipped file.
    directory_to_extract_to
        Path to which the extracted files will be written.

    """
    if not os.path.exists(path):
        logger.info("File does not exist")
    else:
        logger.info(f"Unzipping {path}")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        delete_file(path)


def download_unzip(url, path):
    """Download a zip file and extract its content.

    Parameters
    ----------
    url
        URL from which the the data will be downloaded.
    path
        Path to which the extracted files will be written.

    Notes
    -----
    The downloaded file is assumed to be a ``.zip`` file.

    """
    zip_filepath = f"{path}.zip"
    download_file(url, zip_filepath)
    unzip_file(zip_filepath, path)
