
poetry run python -m srl.train \
        --run-eval-after-training \
        configs/srl/ytvis2021.yaml

# inference and continue training
#poetry run python -m srl.train \
#        --continue [checkpoint_path] \
#        --run-eval-after-training \
#        configs/srl/ytvis2021.yaml