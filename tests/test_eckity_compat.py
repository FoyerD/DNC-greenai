import ast
from importlib.metadata import version
from pathlib import Path


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
    from eckity_dnc import GAIntegerStringVectorCreator

    creator = GAIntegerStringVectorCreator(length=2, bounds=(0, 1))
    individual = creator.individual_from_vector([0, 1])
    assert individual.fitness.higher_is_better


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
