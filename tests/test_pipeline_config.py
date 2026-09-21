import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ModelConfigTests(unittest.TestCase):
    def test_registry_contains_popular_aliases(self):
        model_module = load_module("model_config", SRC_ROOT / "model_config.py")

        self.assertIn("llama-3.1-8b", model_module.SUPPORTED_MODELS)
        self.assertIn("mistral-7b", model_module.SUPPORTED_MODELS)
        self.assertIn("qwen2.5-7b", model_module.SUPPORTED_MODELS)
        self.assertIn("gemma-2-9b", model_module.SUPPORTED_MODELS)

    def test_resolve_model_name_returns_hf_id_for_alias(self):
        model_module = load_module("model_config", SRC_ROOT / "model_config.py")

        self.assertTrue(model_module.resolve_model_name("llama-3.1-8b").endswith("Meta-Llama-3.1-8B-bnb-4bit"))
        self.assertTrue(model_module.resolve_model_name("mistral-7b").endswith("Mistral-7B-v0.3-bnb-4bit"))


class DatasetValidationTests(unittest.TestCase):
    def test_validator_splits_valid_and_invalid_rows(self):
        fake_datasets = types.ModuleType("datasets")

        class FakeDataset(list):
            @classmethod
            def from_list(cls, rows):
                return cls(rows)

        fake_datasets.Dataset = FakeDataset
        fake_datasets.load_dataset = lambda *args, **kwargs: FakeDataset()
        fake_datasets.concatenate_datasets = lambda datasets: FakeDataset([item for ds in datasets for item in ds])
        sys.modules["datasets"] = fake_datasets

        dataset_module = load_module("prepare_dataset", SRC_ROOT / "prepare_dataset.py")

        rows = [
            {"instruction": "Say hi", "input": "", "output": "Hi!"},
            {"instruction": "Missing output", "input": "", "output": ""},
            {"instruction": "Long output", "input": "", "output": "x" * 20000},
            {"instruction": "Dup", "input": "", "output": "Dup"},
            {"instruction": "Dup", "input": "", "output": "Dup"},
        ]

        valid, invalid = dataset_module.validate_rows(rows)

        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 3)

        invalid_codes = {issue["code"] for issue in invalid}
        self.assertIn("missing_output", invalid_codes)
        self.assertIn("duplicate_row", invalid_codes)
        self.assertIn("too_long", invalid_codes)


if __name__ == "__main__":
    unittest.main()
