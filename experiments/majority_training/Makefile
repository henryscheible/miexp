.PHONY: train

train:
	python3 train_transformer.py --device mps --lr 0.01 --dataset_size 1000 --func_width 20 --hidden_dim 2 --head_dim 1 --visualization_save_path ./vis.pdf --csv_save_path ./results.csv --model_save_path ./model.pth --dataset_save_path ./dataset.pt --num_epochs 300 --train_frac 0.4