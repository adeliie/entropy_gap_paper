# The Cross-Entropy Gap: Scaling Laws for Sign Descent and Gradient Descent

Install package `pip install -e .`
Basic libraries are installed as dependencies.
Pytorch needs to be installed separately, see https://pytorch.org/get-started/locally/.




## Get real-data frequencies files
Download ... and put them in the `freqs` subfolder 

## Run the data generation process

```
# Fast scripts (<1min)

# For Figures [...]
python scripts/data_generation/generation_intro.py
# For Figures [...]
python scripts/data_generation/generation_data_sgd.py

# Medium scripts (<4h)

# For Figures [...]
python scripts/data_generation/generation_data_sgd_big.py
# For Figures [...]
python scripts/data_generation/generation_data_gd_opteta.py
# For Figures [...]
python scripts/data_generation/generation_data_sgd_real_data.py

# Slow scripts (<12h)

# For Figures [...]
python scripts/data_generation/generation_data_gd_matrix.py
```

## Make plots
