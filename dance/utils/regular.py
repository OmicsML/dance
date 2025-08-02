import re


def spilt_web(url: str):
    """Parse Weights & Biases URL to extract entity, project and sweep components.

    Parameters
    ----------
    url : str
        Complete wandb sweep URL

    Returns
    -------
    tuple or None
        Tuple of (entity, project, sweep_id) if parsing succeeds
        None if parsing fails

    """
    pattern = r"https://wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/]+)"

    match = re.search(pattern, url)

    if match:
        entity = match.group(1)
        project = match.group(2)
        pattern = r'/sweeps/([^/?]+)'  # Regular expression pattern
        match = re.search(pattern, url)
        if match:
            sweep_id = match.group(1)
            return entity, project, sweep_id
        return None
    else:
        print(url)
        print("No match found")
