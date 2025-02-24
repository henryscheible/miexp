# Profile Transformer Training

## Summary

Runs cProfile on train_transformer_fourier, doesn't extract very many meaningful conclusions. The best place to increase speed would probably be to decrease the number of dataloader intialization calls (currently at ~16 per epoch because of how many different evaluation modes we are using.

More important improvement is the bulk_multiprocessing_experiment.py which parallelizes the existing bulk training loop in experiments/fourier_training

## Usage
* `make`/`make all`/`make visualize`: profiles `train_transformer_fourier` and visualizes the results (using snakeviz)
* `make profile_results.txt`: generates profile without visualiation 
* `make bulk_multiprocessing_experiment`: Runs a small instance of train_bulk (as in fourier_training) but dispatches each training job to a multiprocessing pool, then collates all of the results at the end.