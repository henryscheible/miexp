.PHONY: train

train:
	uv run python train_transformer.py --device cpu \
	--lr 0.01 --wd 1e-1 --batch_size 128 --dataset_size 5000 \
	--func_width 20 --random_frac 0.3 --head_dim 20 \
	--visualization_save_path ./vis/ --csv_save_path ./results.csv --model_save_path ./model.pth --dataset_save_path ./dataset.pt \
	--num_epochs 1000 --train_frac 0.8

visualize:
	uv run python create_visualizations.py 
