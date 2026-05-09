import jax
import jax.numpy as jnp


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
