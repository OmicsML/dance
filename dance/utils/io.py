import pandas as pd
import yaml

from dance.typing import Dict, Optional, PathLike


def load_data_url_dict_from_csv(path: PathLike) -> Dict[str, str]:
    """Load data url dictionary from a two column csv file."""
    df = pd.read_csv(path, header=None).astype(str)
    if df.shape[1] != 2:
        raise ValueError("url metadata csv file must only contain two columns")
    url_dict = {i: j for i, j in df.astype(str).values}
    return url_dict


def read_conditional_parameter(path: PathLike = 'dance/conditional_parameter.yml',
                               conditional_parameter: Optional[str] = None, encoding='utf-8'):
    with open(path, encoding=encoding) as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        if conditional_parameter is None:
            return result
        else:
            return result[conditional_parameter]
