import json
import os
import pathlib

import dropbox
import numpy as np
import pandas as pd
import scanpy as sc
from dropbox.exceptions import ApiError, AuthError

from dance.utils import logger


def upload_file_to_dropbox(dropbox_path, access_token, local_path):
    dbx = dropbox.Dropbox(access_token)

    # Verify access token
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        print("ERROR: Invalid access token; please check your access token.")
        return None
    try:
        file_upload(dbx=dbx, local_path=local_path, remote_path=dropbox_path)
        print("Upload successful.")
    except ApiError as err:
        print(f"API error: {err}")
        return None


def file_upload(dbx: dropbox.Dropbox, local_path: pathlib.Path, remote_path: str):
    CHUNKSIZE = 100 * 1024 * 1024
    upload_session_start_result = dbx.files_upload_session_start(b'')
    cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id, offset=0)
    with local_path.open("rb") as f:
        while True:
            data = f.read(CHUNKSIZE)
            if data == b"":
                break
            logger.debug("Pushing %d bytes", len(data))
            dbx.files_upload_session_append_v2(data, cursor)
            cursor.offset += len(data)
    commit = dropbox.files.CommitInfo(path=remote_path)
    dbx.files_upload_session_finish(b'', cursor, commit)


def create_shared_link(dbx, dropbox_path):
    """Create or get existing shared link.

    :param dbx: Dropbox object
    :param dropbox_path: File path on Dropbox
    :return: Shared link URL

    """
    try:
        links = dbx.sharing_list_shared_links(path=dropbox_path, direct_only=True).links
        if links:
            # If shared link already exists, return the first one
            return links[0].url
        else:
            # Create a new shared link
            link = dbx.sharing_create_shared_link_with_settings(dropbox_path)
            return link.url
    except ApiError as err:
        print(f"Error creating shared link: {err}")
        return None


def get_link(data_fname, local_path, ACCESS_TOKEN, DROPBOX_DEST_PATH):
    DROPBOX_DEST_PATH = DROPBOX_DEST_PATH + "/" + data_fname

    upload_file_to_dropbox(dropbox_path=DROPBOX_DEST_PATH, access_token=ACCESS_TOKEN, local_path=local_path)

    # Create Dropbox object to get shared link
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    # Get shared link
    shared_link = create_shared_link(dbx, DROPBOX_DEST_PATH)
    if shared_link:
        # Dropbox shared link defaults to `dl=0` at the end, which means preview in browser.
        # change it to `dl=1`.
        download_link = shared_link.replace('&dl=0', '&dl=1')
        print(f"Download link: {download_link}")
        return download_link
    else:
        print("Unable to get shared link.")


def get_ans(data: sc.AnnData, tissue: str, dataset_id: str, local_path, ACCESS_TOKEN, DROPBOX_DEST_PATH):
    # keys=["species","tissue","dataset","split","celltype_fname","celltype_url","data_fname","data_url"]
    ans = {}
    ans["species"] = "human"
    ans["tissue"] = tissue.capitalize()
    ans["dataset"] = data.n_obs
    ans["split"] = "train"
    ans["celltype_fname"] = ""
    ans["celltype_url"] = ""
    ans["data_fname"] = f"train_human_{tissue.capitalize()}{dataset_id}_data.h5ad"
    ans["data_url"] = get_link(data_fname=ans["data_fname"].split("_", 1)[1], local_path=local_path,
                               ACCESS_TOKEN=ACCESS_TOKEN, DROPBOX_DEST_PATH=DROPBOX_DEST_PATH)
    ans["is_ALL_Integer"] = np.all(np.equal(data.X.data, data.X.data.astype(int)))
    return ans
