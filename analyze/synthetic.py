import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the parameters for the synthetic time series data
np.random.seed(42)
time_steps = 300  # Total time steps
period = 60  # Period of the sine wave
amplitude = 5  # Amplitude of the sine wave
noise_level = 0.9  # Noise level


time = np.arange(time_steps)

strong_periodic_signal = amplitude * np.sin(2 * np.pi * time / period)
weak_periodic_signal = 2 * np.sin(2 * np.pi * time / (period * 2))
noise = noise_level * np.random.randn(time_steps)
synthetic_series = strong_periodic_signal + weak_periodic_signal + noise
df_futures = pd.DataFrame({
    'Time': time,
    'FuturesPrice': synthetic_series
})
np.random.seed(42)
series_2 = strong_periodic_signal + weak_periodic_signal + noise_level * np.random.randn(time_steps)
series_3 = strong_periodic_signal + weak_periodic_signal + noise_level * np.random.randn(time_steps)

df_futures_multi = pd.DataFrame({
    'Time': time,
    'Series1': synthetic_series,
    'Series2': series_2,
    'Series3': series_3
})

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Series 1
sns.lineplot(x='Time', y='Series1', data=df_futures_multi, ax=axes[0], color='b', lw=2)
axes[0].axis('off')  # Turn off the axes

# Series 2
sns.lineplot(x='Time', y='Series2', data=df_futures_multi, ax=axes[1],  lw=2)
axes[1].axis('off')  # Turn off the axes

# Series 3
sns.lineplot(x='Time', y='Series3', data=df_futures_multi, ax=axes[2], color='g', lw=2)
axes[2].axis('off')  # Turn off the axes

plt.tight_layout()
plt.show()