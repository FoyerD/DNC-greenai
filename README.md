# Deep Neural Crossover for EC-KitY

`eckity-dnc` provides the Deep Neural Crossover (DNC) genetic operator for [EC-KitY](https://github.com/EC-KitY/EC-KitY).

DNC is described in **“Deep Neural Crossover: A Multi-Parent Operator That Leverages Gene Correlations”** ([paper](https://doi.org/10.1145/3638529.3654020)).

## Installation

```bash
pip install eckity-dnc
```

Installing `eckity-dnc` also installs its EC-KitY, PyTorch, NumPy, and SciPy dependencies.

## Usage

Import the public API from `eckity_dnc`:

```python
from eckity_dnc import (
    DeepNeuralCrossover,
    DeepNeuralCrossoverConfig,
    GAIntegerStringVectorCreator,
)
```

Create the vector creator and DNC operator:

```python
population_size = 100
individual_length = 160
number_of_gene_values = 161

individual_creator = GAIntegerStringVectorCreator(
    length=individual_length,
    bounds=(0, number_of_gene_values - 1),
)

dnc_config = DeepNeuralCrossoverConfig(
    embedding_dim=64,
    sequence_length=individual_length,
    num_embeddings=number_of_gene_values,
    running_mean_decay=0.95,
    batch_size=1024,
    learning_rate=1e-4,
    use_device="cpu",
    n_parents=2,
    epsilon_greedy=0.3,
)

dnc_operator = DeepNeuralCrossover(
    probability=0.8,
    population_size=population_size,
    dnc_config=dnc_config,
    individual_evaluator=your_eckity_evaluator,
    vector_creator=individual_creator,
)
```

Add `dnc_operator` to the EC-KitY subpopulation's `operators_sequence`. The supplied evaluator must inherit from EC-KitY's `SimpleIndividualEvaluator` and evaluate vectors created by `individual_creator`.

See [`dnc_runner_eckity.py`](dnc_runner_eckity.py) for a complete bin-packing example using tournament selection and mutation.

## Compatibility

- Python 3.9 or newer
- EC-KitY 0.4.x (tested with 0.4.1)
- PyTorch 2.7.1 or newer (tested with 2.7.1)
- NumPy 2.0.2 or newer (tested with 2.0.2)
- SciPy 1.13.0 or newer (tested with 1.13.0)

## Development

Install the package and development tools, then run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Release preparation and manual TestPyPI/PyPI commands are documented in [`RELEASING.md`](RELEASING.md).

## License

This project is licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
