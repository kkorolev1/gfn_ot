"""Swendsen-Wang sampling and cached Ising evaluation references."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm.auto import trange


def sample_reference(env, beta, num_samples, seed=0, num_chains=128, burn_in=65536, thinning=1024):
    """Approximate the zero-field Ising law with independent cluster chains.

    Bond probability is 1-exp(-2*beta*J). Connected components get independent
    fair binary spins. A block-diagonal graph handles all chains in one call.
    """
    if num_samples < 1 or num_chains < 1 or burn_in < 0 or thinning < 1:
        raise ValueError("Invalid reference sample count or MCMC settings")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Reference beta must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    chains = min(num_chains, num_samples)
    size, n = env.lattice_size, env.max_length
    sites = np.arange(n).reshape(size, size)
    sources = np.tile(sites.reshape(-1), 2)
    destinations = np.concatenate((np.roll(sites, -1, axis=0).reshape(-1),
                                   np.roll(sites, -1, axis=1).reshape(-1)))
    offsets = np.arange(chains)[:, None] * n
    global_sources = sources[None, :] + offsets
    global_destinations = destinations[None, :] + offsets
    states = rng.integers(0, 2, size=(chains, n), dtype=np.int8)
    probability = -np.expm1(-2 * beta * env.coupling)
    collections = (num_samples + chains - 1) // chains
    samples = []
    steps = burn_in + collections * thinning
    for step in trange(steps, desc=f"Ising reference beta={beta:g}", disable=steps < 1000):
        active = ((states[:, sources] == states[:, destinations])
                  & (rng.random((chains, 2 * n)) < probability))
        rows, cols = global_sources[active], global_destinations[active]
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                           shape=(chains * n, chains * n)).tocsr()
        count, labels = connected_components(graph, directed=False)
        states = rng.integers(0, 2, size=count, dtype=np.int8)[labels].reshape(chains, n)
        if step >= burn_in and (step - burn_in + 1) % thinning == 0:
            samples.append(states.copy())
    return np.concatenate(samples, axis=0)[:num_samples]


def get_reference_samples(env, beta, cache_dir, num_samples=16384, seed=0,
                          num_chains=128, burn_in=65536, thinning=1024):
    metadata = dict(version=1, lattice_size=env.lattice_size, coupling=env.coupling,
                    beta=float(beta), num_samples=int(num_samples), seed=int(seed),
                    num_chains=int(num_chains), burn_in=int(burn_in), thinning=int(thinning))
    encoded = json.dumps(metadata, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    cache_dir = Path(cache_dir).expanduser()
    path = cache_dir / f"ising_{env.lattice_size}_beta{beta:g}_{digest}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if str(data['metadata'].item()) != encoded:
                raise ValueError(f"Ising reference cache metadata mismatch: {path}")
            samples = data['samples']
        if samples.shape != (num_samples, env.max_length) or not np.all((samples == 0) | (samples == 1)):
            raise ValueError(f"Invalid Ising reference samples in {path}")
        print(f"Loaded Ising reference: {path}")
        return samples.astype(np.int8)

    samples = sample_reference(env, beta, num_samples, seed, num_chains, burn_in, thinning)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, samples=samples, metadata=encoded)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Cached Ising reference: {path}")
    return samples


def get_reference_pair(env, reference_cfg, seed=0):
    """Use the same reference settings and seeds in preparation and training."""
    left = get_reference_samples(env, env.beta_left, seed=seed + 1729, **dict(reference_cfg))
    right = get_reference_samples(env, env.beta_right, seed=seed + 1730, **dict(reference_cfg))
    return left, right


def main(argv=None):
    """Prepare both evaluation pools without creating a model or training run."""
    import argparse
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    parser = argparse.ArgumentParser(description=(
        'Generate cached Ising MCMC references at both temperatures. '
        'Uses prefix_tb_ising and accepts the same Hydra overrides as run.py.'))
    parser.add_argument('overrides', nargs='*', help='For example: seed=0 reference.cache_dir=.cache/ising')
    args = parser.parse_args(argv)
    config_dir = str(Path(__file__).resolve().parents[2] / 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='base_config', overrides=['algorithm=prefix_tb_ising', *args.overrides])
    if cfg.name != 'prefix_tb_ising':
        parser.error('Reference preparation requires algorithm=prefix_tb_ising')
    left, right = get_reference_pair(instantiate(cfg.env), cfg.reference, cfg.seed)
    print(f"Ising references ready: {len(left)} source and {len(right)} target samples.")


if __name__ == '__main__':
    main()
