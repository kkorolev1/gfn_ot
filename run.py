import os
from datetime import datetime

import hydra
import jax

import matplotlib

from comet_ml import Experiment
from omegaconf import DictConfig, OmegaConf

from utils.helper import flatten_dict, reset_device_memory
from utils.train_selector import get_train_fn


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg = hydra.utils.instantiate(cfg)

    run_name = f"{cfg.name}_{datetime.now()}_seed{cfg.seed}"
    if len(cfg.comet.exp_name) > 0:
        run_name = f"{cfg.comet.exp_name}_{run_name}"

    print("jax devices:", jax.devices())

    # for plotting high-quality stuff
    matplotlib.use("agg")

    comet_exp = None
    if cfg.use_comet:
        comet_exp = Experiment(**cfg.comet)
        comet_exp.set_name(run_name)
        comet_exp.log_parameters(
            flatten_dict(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        )
    train_fn = get_train_fn(cfg.name)

    try:
        if cfg.use_jit:
            train_fn(cfg, comet_exp)
        else:
            with jax.disable_jit():
                train_fn(cfg, comet_exp)
        if cfg.use_comet:
            comet_exp.log_other("error", None)
            comet_exp.end()

    except Exception as e:
        if cfg.use_comet:
            comet_exp.log_other("error", str(e))
            comet_exp.end()
        reset_device_memory()
        raise e


if __name__ == "__main__":
    main()
