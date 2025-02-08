.PHONY: train

train:
	uv run python train_transformer.py --device mps --lr 0.1 --wd 1e-5 --dataset_size 1000 \
	--func_width 5 --head_dim 3 --num_heads 1 \
	--visualization_save_path ./vis.pdf --csv_save_path ./results.csv --model_save_path ./model.pth \
	--dataset_save_path ./dataset.pt --num_epochs 150 --train_frac 0.8

visualize:
	uv run python create_visualizations.py 