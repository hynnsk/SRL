# From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning (ICLR 2026)
Hyun Seok Seong*</sup>, WonJun Moon*</sup>, Jae-Pil Heo</sup> (*: equal contribution)

## Abstract
> Unsupervised object-centric learning models, particularly slot-based architectures, have shown great promise in decomposing complex scenes. However, their reliance on reconstruction-based training creates a fundamental conflict between the sharp, high-frequency attention maps of the encoder and the spatially consistent but blurry reconstruction maps of the decoder. We identify that this discrepancy gives rise to a vicious cycle; the noisy feature map from the encoder forces the decoder to average over possibilities and produce even blurrier outputs, while the gradient computed from blurry reconstruction maps lacks high-frequency details necessary to supervise encoder features. To break this cycle, we introduce Synergistic Representation Learning (SRL) that establishes a virtuous cycle where the encoder and decoder mutually refine one another. SRL leverages the encoder's sharpness to deblur the semantic boundary within the decoder output, while exploiting the decoder's spatial consistency to denoise the encoder's features. This mutual refinement process is stabilized by a warm-up phase with a slot regularization objective that initially allocates distinct entities per slot. By bridging the representational gap between the encoder and decoder, our approach achieves state-of-the-art results on challenging video object-centric learning benchmarks.
----------

## Setup
First, setup the Python environment.
We use [Poetry](https://python-poetry.org/) following the setup procedure from [SlotContrast](https://github.com/martius-lab/slotcontrast).

```
poetry install
```
### Install Options

- `poetry install -E tensorflow` to be able to convert tensorflow datasets
- `poetry install -E coco` to use coco API
- `poetry install -E notebook` to use jupyter notebook and matplotlib

For convenience, we also provide an ```environment.yml``` file:
```
conda env create -f environment.yml
```

## Prepare datasets
To download the datasets used in this work, please follow the instructions in [data/README.md](data/README.md).

The directory structure should look like:
```
📂 [default data dir]/
├── 📁 movi_c/
│   ├── *.tar
│   ├── *.tar
│   └── ...
├── 📁 movi_e/
│   ├── *.tar
│   ├── *.tar
│   └── ...
└── 📁 ytviw2021_resized/
    ├── *.tar
    ├── *.tar
    └── ...
```

By default, datasets are expected to be located in the `/DATA` directory.
If your dataset is stored elsewhere, you can change the default data path by modifying:
- ```./data/utils.py```
- ```./srl/data/utils.py```



## Training
You can train the model using the following scrips:
```
sh scripts/run_movi_c.sh
sh scripts/run_movi_e.sh
sh scripts/run_ytvis2021.sh
```

Checkpoints, metrics, configuration files, and Tensorboard logs will be saved under:
```
../logs/[experiment_name]/checkpoints/[experiment_group].ckpt
../logs/[experiment_name]/metrics/[experiment_group]/hparams.yaml
../logs/[experiment_name]/settings/[experiment_group].yaml
../logs/[experiment_name]/tb/[experiment_group]/events.out.tfevents.*
```
Here, ```[experiment_name]``` and ```[experiment_group]``` are defined in each config file.

To resume training from a previous run, simply rerun the code with the same ```[experiment_name]``` and ```[experiment_group]```.

## Citation
If you find this project useful, please consider the following citation:
```
@article{seong2026synergistic,
  title={From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning},
  author={Seong, Hyun Seok and Moon, WonJun and Heo, Jae-Pil},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## Acknowledgements
This repository is built based on [SlotContrast](https://github.com/martius-lab/slotcontrast) repository. Thanks for the great work.

## License
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.