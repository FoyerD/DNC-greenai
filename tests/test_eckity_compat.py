import ast
import random
from importlib.metadata import version
from pathlib import Path

from eckity.genetic_operators import IntVectorOnePointMutation

from eckity_dnc import GAIntegerStringVectorCreator


ROOT = Path(__file__).parents[1]
RUNNER_FILES = [
    ROOT / "bpp_runner.py",
    ROOT / "dnc_runner_eckity.py",
    ROOT / "eckity_demo.py",
    ROOT / "eckity_runner.py",
]


def test_release_metadata_and_public_imports():
    from eckity_dnc import (
        DeepNeuralCrossover,
        DeepNeuralCrossoverConfig,
        GAIntegerStringVectorCreator,
    )

    assert version("eckity") == "0.4.2"
    assert version("eckity-dnc") == "0.1.2"
    assert all(
        item is not None
        for item in (
            DeepNeuralCrossover,
            DeepNeuralCrossoverConfig,
            GAIntegerStringVectorCreator,
        )
    )


def test_vector_creator_still_uses_fitness_direction():
    creator = GAIntegerStringVectorCreator(length=2, bounds=(0, 1))
    individual = creator.individual_from_vector([0, 1])
    assert individual.fitness.higher_is_better


def test_runners_use_native_integer_uniform_mutation():
    legacy_class = "IntVector" + "UniformMutation"
    legacy_helper = "uniform_cell_" + "selector"
    source_files = [
        ROOT / "bpp_runner.py",
        ROOT / "dnc_aux.py",
        ROOT / "dnc_runner_eckity.py",
    ]

    for path in source_files:
        source = path.read_text(encoding="utf-8")
        assert legacy_class not in source
        assert legacy_helper not in source
    for path in (ROOT / "bpp_runner.py", ROOT / "dnc_runner_eckity.py"):
        assert "IntVectorOnePointMutation" in path.read_text(encoding="utf-8")

    creator = GAIntegerStringVectorCreator(length=8, bounds=(0, 5))
    original = [0, 1, 2, 3, 4, 5, 0, 1]
    first = creator.individual_from_vector(original.copy())
    second = creator.individual_from_vector(original.copy())
    mutation = IntVectorOnePointMutation(
        probability=1.0,
        probability_for_each=1.0,
    )

    assert mutation.cell_selector(first) == list(range(first.size()))
    random.seed(7)
    succeeded, _ = mutation.attempt_operator([first], 0)
    random.seed(7)
    mutation.attempt_operator([second], 0)

    assert succeeded
    assert first.vector == second.vector
    assert first.vector != original
    assert all(0 <= value <= 5 for value in first.vector)


def test_runners_do_not_use_legacy_selection_tuples_or_direction_keywords():
    for path in RUNNER_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert not (
                call_name in {"TournamentSelection", "ElitismSelection"}
                and any(keyword.arg == "higher_is_better" for keyword in node.keywords)
            )
            for keyword in node.keywords:
                if keyword.arg == "selection_methods" and isinstance(
                    keyword.value, ast.List
                ):
                    assert all(
                        not isinstance(item, ast.Tuple) for item in keyword.value.elts
                    )
