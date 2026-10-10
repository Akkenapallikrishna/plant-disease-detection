from conftest import FAKE_CLASSES, run_prepare
from prepare_data import split_counts


def test_split_counts_add_up():
    assert split_counts(100, 0.70, 0.15) == (70, 15, 15)
    for n in (10, 37, 100):
        assert sum(split_counts(n, 0.70, 0.15)) == n


def test_balanced_split_and_nested_duplicate_skipped(fake_data_dir):
    for split, expected in (("train", 7), ("val", 2), ("test", 1)):
        class_dirs = sorted(p.name for p in (fake_data_dir / split).iterdir())
        assert class_dirs == sorted(FAKE_CLASSES), "nested 'PlantVillage' folder must not become a class"
        for name in FAKE_CLASSES:
            assert len(list((fake_data_dir / split / name).iterdir())) == expected


def test_no_image_appears_in_two_splits(fake_data_dir):
    seen = {}
    for split in ("train", "val", "test"):
        for img in (fake_data_dir / split).rglob("*.JPG"):
            key = (img.parent.name, img.name)
            assert key not in seen, f"{key} is in both {seen.get(key)} and {split} (data leakage)"
            seen[key] = split


def test_same_seed_gives_same_sample(fake_source, tmp_path):
    run_prepare(fake_source, tmp_path / "a")
    run_prepare(fake_source, tmp_path / "b")
    manifest_a = (tmp_path / "a" / "split_manifest.csv").read_text()
    manifest_b = (tmp_path / "b" / "split_manifest.csv").read_text()
    assert manifest_a == manifest_b


def test_fails_when_a_class_has_too_few_images(fake_source, tmp_path):
    result = run_prepare(fake_source, tmp_path / "out", per_class=50)
    assert result.returncode != 0
