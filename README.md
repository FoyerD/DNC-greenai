# Deep Neural Crossover (DNC)

This repository implements the **Deep Neural Crossover (DNC)** operator as described in the paper:  
**"Deep Neural Crossover: A Multi-Parent Operator That Leverages Gene Correlations"**  
[ACM Link](https://dl.acm.org/doi/abs/10.1145/3638529.3654020)
---


## 🚀 Quick Start

To run a full example using the DNC operator on the **Bin Packing Problem**, execute:

```bash
python dnc_runner_eckity.py
```

This example uses:
- Integer vector representation of solutions.
- A domain-specific fitness evaluator for bin packing.
- Tournament selection, and n-point mutation.

---

## 📁 File Structure

- `dnc_runner_eckity.py` - Main script running the evolutionary algorithm with the DNC operator.
- `datasets/hard_parsed.json` - A JSON file containing bin packing datasets used in the paper.
- Custom components:
  - `BinPackingEvaluator` - Evaluates fitness based on bin packing constraints.
---

## Customization Guide

To use this code for a different problem domain, **only a few parts need to be modified**:

### 1. Define a Custom Evaluator

Replace `BinPackingEvaluator` with a custom class that inherits from `SimpleIndividualEvaluator`:

```python
class YourEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        # return a float indicating fitness
        return your_fitness_function(individual.vector)
```

### 2. Update the Creator (if needed)

If you need special initialization, adjust `GAIntegerStringVectorCreator`.

### 3. Configure DNC

The DNC operator requires setting the correct number of embedding dimensions, which should correspond to the **number of distinct values each gene can take** (i.e., the number of bins, colors, etc.).

```python
DeepNeuralCrossoverConfig(
    gene_value_dim=MAX_GENE_VALUE + 1,
    embedding_dim=EMBED_DIM,  # e.g., 64
    ...
)
```

---

## ⚙️ Evolutionary Parameters

You can easily adjust standard evolutionary settings:

```python
population_size = 100
crossover_rate = 0.5
mutation_rate = 0.1
num_generations = 6000
```

These are passed to the `Subpopulation` and `SimpleEvolution` constructors.

---