"""Download the CDC codebook page of every 2005-2018 data file that holds a feature, and pull out
each variable's value table.

  codebooks/<FILE>.htm     raw pages (cache; re-runs skip files already there)
  codebook_values.csv      module, cycle, variable, code, description  (one row per table row;
                           a range such as "1 to 104" is kept as written)
"""
import re
import subprocess
import sys
import time
from html import unescape
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import DATA, WORK  # noqa: E402

PAGES = WORK / "codebooks"
YEAR = {"_D": 2005, "_E": 2007, "_F": 2009, "_G": 2011, "_H": 2013, "_I": 2015, "_J": 2017}
SHORT = {"_D": "2005-06", "_E": "2007-08", "_F": "2009-10", "_G": "2011-12", "_H": "2013-14",
         "_I": "2015-16", "_J": "2017-18"}
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.htm"
VAR = re.compile(r'<h3 class="vartitle" id="([^"]+)"')
ROW = re.compile(r'<tr>\s*<td scope="row" class="values">(.*?)</td>\s*<td class="values">(.*?)</td>', re.S)


def files_needed():
    names = {p.stem.upper() for p in DATA.glob("*/*") if p.suffix.lower() == ".xpt"}
    cat = pd.read_csv(WORK / "feature_catalog.csv")
    modules = set(cat[cat.is_feature | cat.category.eq("renamed")].module)
    return sorted(n for n in names if n[-2:] in YEAR and n[:-2] in modules)


def fetch(name):
    out = PAGES / f"{name}.htm"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    url = URL.format(year=YEAR[name[-2:]], name=name)
    subprocess.run(["curl", "-s", "-m", "60", "-o", str(out), url], check=True)
    time.sleep(1)
    return out


def parse(path):
    html = path.read_text(encoding="utf-8", errors="replace")
    heads = [(m.start(), m.group(1)) for m in VAR.finditer(html)] + [(len(html), None)]
    for (start, var), (end, _) in zip(heads, heads[1:]):
        for code, desc in ROW.findall(html[start:end]):
            yield var.upper(), unescape(code).strip(), unescape(re.sub(r"\s+", " ", desc)).strip()


def main():
    PAGES.mkdir(exist_ok=True)
    rows = []
    for name in files_needed():
        path = fetch(name)
        got = list(parse(path))
        if not got:
            print("no tables:", name)
        rows += [(name[:-2], SHORT[name[-2:]], v, c, d) for v, c, d in got]
    pd.DataFrame(rows, columns=["module", "cycle", "variable", "code", "description"]).to_csv(
        WORK / "codebook_values.csv", index=False)
    print(len(rows), "value rows from", len({(r[0], r[1]) for r in rows}), "files")


if __name__ == "__main__":
    main()
