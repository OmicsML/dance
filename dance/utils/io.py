import pandas as pd

from dance.typing import Dict, PathLike


def load_data_url_dict_from_csv(path: PathLike) -> Dict[str, str]:
    """Load data url dictionary from a two column csv file."""
    df = pd.read_csv(path, header=None).astype(str)
    if df.shape[1] != 2:
        raise ValueError("url metadata csv file must only contain two columns")
    url_dict = {i: j for i, j in df.astype(str).values}
    return url_dict
