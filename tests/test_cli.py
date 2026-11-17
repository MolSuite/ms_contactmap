"""The command line finds the ligand on its own and exports by suffix."""
from __future__ import annotations

from pathlib import Path

import pytest

from ms_contactmap.__main__ import LigandError, find_site, main

DATA = Path(__file__).resolve().parents[1] / "data"


def test_ligand_is_found_narrowed_or_refused(tmp_path, capsys):
    assert find_site(DATA / "4uwh.pdb") == ("JXM", "A", 1876)
    # Copies of one ligand: batch draws the first and says so.
    assert find_site(DATA / "4ps5.pdb") == ("2TA", "A", 201)
    assert "copies of 2TA" in capsys.readouterr().err
    assert find_site(DATA / "4ps5.pdb", chain="B") == ("2TA", "B", 201)
    # Different ligands: batch refuses, a window asks.
    with pytest.raises(LigandError, match="several ligands"):
        find_site(DATA / "6wak.pdb")
    assert find_site(DATA / "6wak.pdb", "TQA") == ("TQA", "C", 1301)
    assert find_site(DATA / "6wak.pdb", ask=lambda sites: sites[-1]) == ("TQA", "C", 1301)
    # No small molecule: protein-protein and friends are not supported yet.
    apo = tmp_path / "apo.pdb"
    apo.write_text("".join(line for line in (DATA / "4uwh.pdb").open() if line.startswith("ATOM")))
    with pytest.raises(LigandError, match="not supported yet"):
        find_site(apo)
    # A name the user gives is trusted even when it looks like an additive.
    assert find_site(apo, "GOL") == ("GOL", None, None)


def test_batch_export_by_suffix(tmp_path):
    svg, json_file = tmp_path / "d.svg", tmp_path / "d.json"
    assert main([str(DATA / "4uwh.pdb"), "--no-exposure", "-o", str(svg), "-o", str(json_file)]) == 0
    assert svg.read_text().lstrip().startswith("<?xml") or "<svg" in svg.read_text()
    assert json_file.stat().st_size > 0
    with pytest.raises(SystemExit):
        main([str(DATA / "4uwh.pdb"), "-o", str(tmp_path / "d.pdf")])
    with pytest.raises(SystemExit):
        main([str(DATA / "4uwh.pdb"), "--gui", "diagram,4d"])
