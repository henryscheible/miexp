.PHONY: train

train:
	uv run python train_transformer.py --device cpu --lr 0.1 --wd 1e-5 --dataset_size 1000 \
	--func_width 20 --head_dim 32 --num_heads 16 --hidden_dim 32 \
	--visualization_save_path ./vis.pdf --csv_save_path ./results.csv --model_save_path ./model.pth \
	--dataset_save_path ./dataset.pt --num_epochs 300 --train_frac 0.8

visualize:
	uv run python create_visualizations.py 
