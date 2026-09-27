import jax
import jax.numpy as jnp
import numpy as np


def build_evaluation_buffer(target_dist, capacity, seed):
    """Keep the latest terminal samples and their histogram for TV evaluation.

    Each update appends one batch, evicting the oldest samples once full.
    Store flattened state indices on the host and update counts incrementally.
    Return (None, {}) until the buffer contains at least one sample.
    The startup baseline compares two independent target subsets of capacity
    samples each; model TV compares the buffered histogram to the true target.
    """
    if capacity < 1:
        raise ValueError("Evaluation buffer capacity must be positive")
    target_dist = np.asarray(target_dist, dtype=np.float64)
    probs = target_dist.reshape(-1)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    # fmt: off
    target_histogram = np.bincount(rng.choice(len(probs), size=capacity, p=probs), minlength=len(probs)) / capacity
    baseline = {"tv": float(0.5 * np.abs(probs - target_histogram).sum())}

    buffer = np.empty(capacity, dtype=np.int64)
    counts = np.zeros(len(probs), dtype=np.int64)
    size = cursor = 0

    def update(samples):
        nonlocal size, cursor, counts
        samples = np.asarray(samples, dtype=np.int32)
        indices = np.ravel_multi_index(samples.T, target_dist.shape)
        if len(indices) >= capacity:
            buffer[:] = indices[-capacity:]
            counts = np.bincount(buffer, minlength=len(probs))
            size, cursor = capacity, 0
        else:
            positions = (cursor + np.arange(len(indices))) % capacity
            # Before the first wrap, only buffer[:size] has been populated.
            evicted = buffer[positions[positions < size]]
            counts -= np.bincount(evicted, minlength=len(probs))
            buffer[positions] = indices
            counts += np.bincount(indices, minlength=len(probs))
            size = min(capacity, size + len(indices))
            cursor = (cursor + len(indices)) % capacity
        if size == 0:
            return None, {}
        empirical = counts / size
        metrics = {"tv": float(0.5 * np.abs(probs - empirical).sum())}
        return empirical.reshape(target_dist.shape), metrics

    return update, baseline


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key for the current level of the dictionary.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reset_device_memory(delete_objs=True):
    """Free all tracked DeviceArray memory and delete objects.
    Args:
      delete_objs: bool: whether to delete all live DeviceValues or just free.
    Returns:
      number of DeviceArrays that were manually freed.
    """
    # https://github.com/google/jax/issues/1222#issuecomment-597683078
    backend = jax.lib.xla_bridge.get_backend()  # type: ignore
    for buf in backend.live_buffers():
        buf.delete()
    return None


def extract_last_entry(dictionary):
    last_entries = {}
    for key, value in dictionary.items():
        try:
            last_entries[key] = value[-min(len(value), 1)]
        except:
            pass
    return last_entries


@jax.custom_derivatives.custom_jvp
@jax.jit
def log1mexp(x):
    r"""Numerically stable calculation of :math:`\log(1 - \exp(-x))`.

    This function is undefined for :math:`x < 0`.

    Based on `TensorFlow's implementation <https://www.tensorflow.org/probability/api_docs/python/tfp/math/log1mexp>`_.

    References:
      .. [1] Martin Mächler. `Accurately Computing log(1 − exp(−|a|)) Assessed by the Rmpfr package.
        <https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf>`_.
    """
    c = jnp.log(2.0)
    return jnp.where(
        x < c,
        jnp.log(-jnp.expm1(-x)),
        jnp.log1p(-jnp.exp(-x)),
    )


log1mexp.defjvps(lambda g, ans, x: g / jnp.expm1(x))
