def get_train_fn(alg_name):
    if alg_name == "prefix_tb_hypergrid":
        from envs.hypergrid.prefix_tb_hypergrid import prefix_tb_hypergrid_trainer

        return prefix_tb_hypergrid_trainer
    else:
        raise ValueError(f"Unknown algorithm {alg_name}.")
