import hydra
from omegaconf import OmegaConf

def set_config(confg_name):
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    config_path = '../env'

    hydra.initialize(config_path=config_path)

    cfg = hydra.compose(config_name=confg_name, overrides=[])

    return cfg