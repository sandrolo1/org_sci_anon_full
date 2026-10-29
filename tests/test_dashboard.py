import json
import re

import yaml

from research0.dashboard import render


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor("!", lambda loader, suffix, node: None)


def test_render_escapes_hostile_text_and_round_trips(tmp_path):
    snapshot = {"title": "</script><img src=x onerror=alert(1)>", "items": [], "note": " "}
    page = render(snapshot, tmp_path).read_text(encoding="utf-8")
    data = re.search(r'<script type="application/json" id="snapshot">(.*?)</script>', page, re.S).group(1)
    assert "</script><img" not in page and "<img" not in data
    assert json.loads(data) == snapshot
    assert json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8")) == snapshot
