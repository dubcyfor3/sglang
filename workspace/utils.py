import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def set_block_size_and_threshold(config_path, block_size, threshold):
    """
    Set the block size and confidence threshold in a YAML config file.
    
    Args:
        config_path: Path to the YAML config file
        block_size: The block size value to set
        threshold: The confidence threshold value to set (0.0 - 1.0)
    """
    config = load_config(config_path)
    config['block_size'] = block_size
    config['threshold'] = threshold
    print(f"Block size: {block_size}, Confidence threshold: {threshold}")
    save_config(config, config_path)
    print(f"Config saved to {config_path}")