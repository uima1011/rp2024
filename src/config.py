import yaml

def load(config_path="src/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
