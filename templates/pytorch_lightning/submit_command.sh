grid run \
--instance_type 2_m60_8gb \
--framework lightning \
--gpus 2 \
tune_classification_basic.py \
--learning_rate "uniform(1e-5, 1e-1, 5)" \
--batch_size "[64, 128, 256]"
