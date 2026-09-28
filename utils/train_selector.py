def get_train_fn(alg_name):
    if alg_name == "prefix_tb_hypergrid":
        from envs.hypergrid.prefix_tb_hypergrid import prefix_tb_hypergrid_trainer

        return prefix_tb_hypergrid_trainer
    elif alg_name == "prefix_tb_tfbind":
        from envs.tfbind.prefix_tb_tfbind import prefix_tb_tfbind_trainer

        return prefix_tb_tfbind_trainer
    elif alg_name == "prefix_tb_amp":
        from envs.amp.prefix_tb_amp import prefix_tb_amp_trainer

        return prefix_tb_amp_trainer
    else:
        raise ValueError(f"Unknown algorithm {alg_name}.")
