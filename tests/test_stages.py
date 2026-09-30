import os
import time

from flood.stages import Stage, all_stages


def _stage(tmp_path, calls):
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"

    def run(region):
        calls.append(1)
        dst.write_text(src.read_text())

    return Stage("demo", run, lambda r: [src], lambda r: [dst], ("hazard",)), src


def test_skips_when_up_to_date_and_reruns_on_change(region, tmp_path):
    calls = []
    stage, src = _stage(tmp_path, calls)
    src.write_text("a")
    assert stage.execute(region) is True
    assert stage.execute(region) is False
    assert stage.execute(region, force=True) is True

    later = time.time() + 10
    src.write_text("b")
    os.utime(src, (later, later))
    assert stage.execute(region) is True
    assert len(calls) == 3


def test_reruns_when_settings_change(region, tmp_path):
    stage, src = _stage(tmp_path, [])
    src.write_text("a")
    stage.execute(region)
    region.settings["hazard"]["zones"]["breaks"][1] = 0.25
    fresh, reason = stage.is_up_to_date(region)
    assert not fresh and "settings changed" in reason


def test_pipeline_order():
    names = [s.name for s in all_stages()]
    assert names == [
        "preprocess",
        "terrain",
        "climate",
        "landcover",
        "runoff",
        "hazard",
        "exposure",
        "validation",
        "scenarios",
        "figures",
    ]
