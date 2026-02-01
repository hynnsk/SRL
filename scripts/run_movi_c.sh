
poetry run python -m slotcontrast.train \
        --run-eval-after-training \
        configs/slotcontrast/movi_c.yaml

# inference and continue training
#poetry run python -m slotcontrast.train \
#        --continue [checkpoint_path] \
#        --run-eval-after-training \
#        configs/slotcontrast/movi_c.yaml
