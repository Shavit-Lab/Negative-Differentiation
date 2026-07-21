# [Negative Pre-activations Differentiate Syntax](https://openreview.net/forum?id=RzcCrU0tXP) - ICLR 2026
Here we include the code for our paper [Negative Pre-activations Differentiate Syntax](https://openreview.net/forum?id=RzcCrU0tXP).

![Example of negative differentiation](assets/pythia_neuron.png)
*A Wasserstein neuron in Pythia 1.4B differentiating similar inputs to two distinct negative values.*

### Environment Setup
1. Create a conda environment:
   ```
   conda env create -f environment.yml
   ```
2. Install pip dependencies:
   ```
   conda activate negative-diff
   pip install -r requirements.txt
   ```

### Cite
If you found our work useful, please cite our paper:
```
@inproceedings{kong2026negative,
  title={Negative Pre-activations Differentiate Syntax},
  author={Kong, Linghao and Ning, Angelina and Adler, Micah and Shavit, Nir N},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
