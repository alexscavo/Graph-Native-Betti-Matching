from pathlib import Path
import tempfile

import pytest

from scripts.relocate_real_graph_dataset import copy_checked, move_checked


def test_graph_copy_is_verified_and_resumable():
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source" / "graph.vvg"
        destination = Path(folder) / "destination" / "graph.vvg"
        source.parent.mkdir()
        source.write_bytes(b"selected full-volume graph")
        copy_checked(source, destination)
        copy_checked(source, destination)
        assert destination.read_bytes() == source.read_bytes()
        destination.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="differs"):
            copy_checked(source, destination)


def test_patch_move_handles_resumption_without_overwriting():
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "mixed" / "patch.vtp"
        destination = Path(folder) / "ixi" / "patch.vtp"
        source.parent.mkdir()
        source.write_bytes(b"real graph patch")
        move_checked(source, destination)
        move_checked(source, destination)
        assert not source.exists()
        assert destination.read_bytes() == b"real graph patch"
        source.write_bytes(b"unexpected duplicate")
        with pytest.raises(FileExistsError, match="source and destination"):
            move_checked(source, destination)
