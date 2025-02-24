all: visualize

profile_results.txt:
	uv run python run_profiler.py

visualize: profile_results.txt
	uv run snakeviz profile_results.txt

bulk_multiprocessing_experiment:
	uv run python bulk_multiprocessing_experiment.py