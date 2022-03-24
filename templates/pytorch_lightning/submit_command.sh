grid run \
--instance_type 2_m60_8gb \
--framework lightning \
--gpus 2 \
tune_classification_basic.py \
--learning_rate "[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]" \
--batch_size "[64, 128, 256]" \
--epochs 20
