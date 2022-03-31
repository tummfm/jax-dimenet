# Jax-DimeNet++

Haiku implementation of the [DimeNet++ architecture](https://github.com/gasteigerjo/dimenet).

## Getting started
This repository provides 2 interfaces for the DimeNet++ model. The first
interface allows using DimeNet++ as a
[Jax M.D.](https://github.com/google/jax-md) *energy_fn* to run molecular
dynamics simulations. The second interface allows prediction of global molecular
properties.

```python
from jax_dimenet import dimenet, sparse_graph

# Jax M.D. energy function:
init_fn, dimenet_energy_fn = dimenet.energy_neighborlist(displacement_fn, r_cut)
init_params = init_fn(random.PRNGKey(0), positions, neighbor=neighbors)
energy_fn = partial(dimenet_energy_fn, init_params)  # jax_md energy_fn interface
print('Predicted energy:', energy_fn(positions, neighbors))

# Molecular property prediction:
mol_graph, _ = sparse_graph.sparse_graph_from_neighborlist(
    displacement_fn, positions, neighbors, r_cut)
init_fn, property_predictor = dimenet.property_prediction(r_cut, n_targets=5)
init_params = init_fn(random.PRNGKey(0), mol_graph)
print('Predicted properties:', property_predictor(init_params, mol_graph))
```

A minimum usage example is available in the [notebooks folder](notebooks/usage_example.ipynb). For 
more real-world applications of the DimeNet++ model in MD simulations, please
refer to the [DiffTRe](https://github.com/tummfm/difftre) repository.

## Installation
You can install Jax-DimeNet via pip:
```
pip install jax-dimenet
```

## Requirements
The repository uses the following packages:
```
    jax>=0.2.12
    jaxlib>=0.1.65
    jax-md>=0.1.13
    dm-haiku>=0.0.4
    sympy
    chex
```
The code was run with Python 3.8.

## Contribution
Contributions are always welcome! Please open a pull request to discuss the code
additions.

## Contact
For questions, please contact stephan.thaler@tum.de.

## Citation
If you use this code in your own work, please consider citing the paper that
introduced this DimeNet++ implementation,
```text
@article{thaler_difftre_2021,
  title = {Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting},
  author = {Thaler, Stephan and Zavadlav, Julija},
  journal={Nature Communications},
  volume={12},
  pages={6884},
  doi={10.1038/s41467-021-27241-4}
  year = {2021}
}
```
as well as the original DimeNet++ paper.
```text
@inproceedings{klicpera_dimenetpp_2020,
  title = {Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules},
  author = {Klicpera, Johannes and Giri, Shankari and Margraf, Johannes T. and G{\"u}nnemann, Stephan},
  booktitle={NeurIPS-W},
  year = {2020}
}
```