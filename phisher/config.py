import tomllib


def config(config_path="./config.toml") -> dict:
    """Read config from toml file

    Args:
        config_path (str, optional): path where config is. Defaults to "./config.toml".

    Returns:
        dict: read cofig values
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


config = config()
