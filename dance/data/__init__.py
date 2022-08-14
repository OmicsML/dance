# Copyright 2022 DSE lab.  All rights reserved.
import os
import urllib.request
import zipfile

import tqdm


# delete zip file
def delete_file(filename):
    if not os.path.exists(filename):
        print("File does not exist")
    else:
        print("Deleting", filename)
        os.remove(filename)


# download zip file from url with progress bar
def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        u = urllib.request.urlopen(url)
        f = open(filename, "wb")
        meta = u.info()
        file_size = int(meta.get("Content-Length"))
        print(f"Downloading: {filename} Bytes: {file_size:,}")

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
        print("File already downloaded")
        return False


# unzip zipfile
def unzip_file(filename, directory_to_extract_to):
    if not os.path.exists(filename):
        print("File does not exist")
    else:
        print("Unzipping", filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        delete_file(filename)


def download_unzip(url, filepath):
    zip_filepath = f"{filepath}.zip"
    download_file(url, zip_filepath)
    unzip_file(zip_filepath, filepath)
