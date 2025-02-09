.PHONY: train all visualize

all: train visualize

train:
	uv run python train_transformer.py --device mps --lr 0.01 --wd 0 --dataset_size 1000 \
	--func_width 10 --head_dim 3 --num_heads 1 --csv_save_path ./results.csv --model_save_path ./model.pth \
	--dataset_save_path ./dataset.pt --num_epochs 150 --train_frac 0.6

visualize:
	uv run python create_visualizations.py 

train_dry_run:
	uv run python train_transformer.py --device cpu --lr 0.01 --wd 0 --dataset_size 10 \
	--func_width 3 --head_dim 2 --num_heads 1 --num_epochs 1 --train_frac 0.6 --random_seed 0

bulk_train:
	uv run python train_bulk.py --device mps --lr 0.01 --wd 0 --dataset_size 500 --func_width 10 \
	--head_dim 3 --num_heads 1 --event_csv_save_path results/bulk_events.csv --metadata_csv_save_path results/bulk_metadata.csv \
	--num_epochs 100 --model_save_path results/bulk_models.csv --train_frac 0.5 --num_components 4 --comp_p 0.3 \
	--num_trials_per_function 5 --num_functions 5