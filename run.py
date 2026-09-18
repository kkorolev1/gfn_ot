import os
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
import jax

import matplotlib

from omegaconf import DictConfig, OmegaConf

from utils.experiment_logger import create_logger
from utils.train_selector import get_train_fn


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run_name = f"{cfg.name}_{datetime.now()}_seed{cfg.seed}"
    if len(cfg.comet.exp_name) > 0:
        run_name = f"{cfg.comet.exp_name}_{run_name}"

    print("jax devices:", jax.devices())

    # for plotting high-quality stuff
    matplotlib.use("agg")

    parameters = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    comet_options = dict(parameters["comet"])
    comet_options.pop("exp_name", None)
    experiment_logger = create_logger(
        cfg.logger, cfg.log_dir or HydraConfig.get().runtime.output_dir,
        run_name=run_name, checkpoint_filename=cfg.checkpoint_filename,
        comet_options=comet_options,
    )
    print("logs and checkpoint:", experiment_logger.log_dir)
    with experiment_logger:
        experiment_logger.log_parameters(parameters)
        train_fn = get_train_fn(cfg.name)
        cfg = hydra.utils.instantiate(cfg)
        if cfg.use_jit:
            train_fn(cfg, experiment_logger)
        else:
            with jax.disable_jit():
                train_fn(cfg, experiment_logger)


if __name__ == "__main__":
    main()
