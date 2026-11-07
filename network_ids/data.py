"""Generate synthetic network-flow-like features; not actual packet captures."""
import numpy as np
import pandas as pd

def generate_flows(n=2000, seed=42):
    if n < 20: raise ValueError('n must be >=20')
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, 0.32, size=n)
    duration = rng.exponential(3, n) + label*rng.exponential(1, n)
    bytes_sent = rng.lognormal(7 + label*0.75, 0.7, n)
    packets = rng.poisson(12 + 15*label, n)
    failed_logins = rng.poisson(0.1 + 2.0*label, n)
    port_entropy = np.clip(rng.normal(0.3 + 0.3*label, 0.12, n), 0, 1)
    return pd.DataFrame({'duration':duration, 'bytes_sent':bytes_sent, 'packets':packets,
                         'failed_logins':failed_logins, 'port_entropy':port_entropy,
                         'label':label})
