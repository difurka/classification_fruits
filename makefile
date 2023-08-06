.PHONY: all train clean get_data load_data test

all: get_data train

train:
	python3 train.py

test:
	python3 test.py

sweep:
	python3 sweep.py


clean:
	rm -rf data
	rm -rf outs
	rm -rf wandb
	rm -rf artifacts
	rm -rf src/__pycache__
	rm -rf src/weights
	rm -rf configs/__pycache__
	rm -rf notebooks/wandb
	rm -rf notebooks/artifacts
	rm -rf notebooks/data

load_data:
	python3 src/load_data_to_wandb.py

get_data:
	python3 src/load_data_from_wandb.py
