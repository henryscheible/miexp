.PHONY: train all visualize

all: train visualize

train:
	uv run python train_transformer.py --device mps --lr 0.01 --wd 0 --dataset_size 1000 \
	--func_width 10 --head_dim 3 --num_heads 1 --csv_save_path ./results.csv --model_save_path ./model.pth \
	--dataset_save_path ./dataset.pt --num_epochs 150 --train_frac 0.6 --random_seed 0

visualize:
	uv run python create_visualizations.py 

train_dry_run:
	uv run python train_transformer.py --device cpu --lr 0.01 --wd 0 --dataset_size 10 \
	--func_width 3 --head_dim 2 --num_heads 1 --num_epochs 1 --train_frac 0.6 --random_seed 0

bulk_train:
	mkdir results && uv run python train_bulk.py --device mps --lr 0.01 --wd 0 --dataset_size 1000 --func_width 10 \
	--head_dim 3 --num_heads 1 --event_csv_save_path results/bulk_events.csv --metadata_csv_save_path results/bulk_metadata.csv --bulk_conf_save_path results/bulk_conf.yaml \
	--num_epochs 200 --model_save_path results/bulk_models.pt --train_frac 0.5 --num_components 4 --comp_p 0.3 \
	--num_trials_per_function 4 --num_functions 5 --init_random_seed 0 --low_reject_threshold 0.4 --high_reject_threshold 0.6

bulk_train_cpu:
	uv run python train_bulk.py --device cpu --lr 0.01 --wd 0 --dataset_size 1000 --func_width 10 \
	--head_dim 3 --num_heads 1 --event_csv_save_path results/bulk_events.csv --metadata_csv_save_path results/bulk_metadata.csv --bulk_conf_save_path results/bulk_conf.yaml \
	--num_epochs 200 --model_save_path results/bulk_models.pt --train_frac 0.5 --num_components 4 --comp_p 0.3 \
	--num_trials_per_function 2 --num_functions 5 --init_random_seed 0 --low_reject_threshold 0.4 --high_reject_threshold 0.6

bulk_train_cuda:
	uv run python train_bulk.py --device cuda --lr 0.01 --wd 0 --dataset_size 1000 --func_width 10 \
	--head_dim 3 --num_heads 1 --event_csv_save_path results/bulk_events.csv --metadata_csv_save_path results/bulk_metadata.csv --bulk_conf_save_path results/bulk_conf.yaml \
	--num_epochs 200 --model_save_path results/bulk_models.pt --train_frac 0.5 --num_components 4 --comp_p 0.3 \
	--num_trials_per_function 2 --num_functions 2 --init_random_seed 0 --low_reject_threshold 0.4 --high_reject_threshold 0.6


bulk_train_dry_run:
	uv run python train_bulk.py --device cpu --lr 0.01 --wd 0 --dataset_size 150 --func_width 5 \
	--head_dim 3 --num_heads 1 --event_csv_save_path results_dry_run/bulk_events.csv --metadata_csv_save_path results_dry_run/bulk_metadata.csv --bulk_conf_save_path results_dry_run/bulk_conf.yaml \
	--num_epochs 20 --model_save_path results_dry_run/bulk_models.pt --train_frac 0.5 --num_components 4 --comp_p 0.3 \
	--num_trials_per_function 1 --num_functions 5 --init_random_seed 0 --low_reject_threshold 0.3 --high_reject_threshold 0.7
