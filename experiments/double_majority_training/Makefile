.PHONY: train

train:
	uv run python train_transformer.py --device mps --lr 0.01 --dataset_size 1000 --func_width 20 --head_dim 1 --visualization_save_path ./vis.pdf --csv_save_path ./results.csv --model_save_path ./model.pth --dataset_save_path ./dataset.pt --num_epochs 300 --train_frac 0.4 --low-threshold 0.5 --high-threshold 

visualize:
	uv run python create_visualizations.py 

train_dry_run: 
	uv run python train_transformer.py --device cpu --lr 0.01 --dataset_size 20 --func_width 3 --head_dim 1 --num_epochs 1 --train_frac 0.4

