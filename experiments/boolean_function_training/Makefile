.PHONY: train

train:
	python3 train_transformer.py --device mps --lr 0.01 --dataset_size 1000 --func_width 20 --hidden_dim 8 --head_dim 8 --visualization_save_path ./vis.pdf --csv_save_path ./results.csv --model_save_path ./model.pth --dataset_save_path ./dataset.pt --num_epochs 100 --train_frac 0.005