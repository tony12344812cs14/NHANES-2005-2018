"""Download the public NHANES files the pipeline reads into <NHANES_DATA>/<cycle>/<NAME>.xpt,
plus the Multum drug-category file RXQ_DRUG.xpt. About 0.6 GB. Files already present are skipped,
so a re-run resumes. Every attempt is recorded in <NHANES_DATA>/manifest_<timestamp>.json.

Usage: python -m nhanes.download
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from nhanes.paths import DATA, DRUG_FILE

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
DRUG_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1988/DataFiles/RXQ_DRUG.xpt"
CYCLES = {  # cycle folder -> (first year in the URL, file suffix or "P_" prefix)
    "2005-2006": (2005, "_D"), "2007-2008": (2007, "_E"), "2009-2010": (2009, "_F"),
    "2011-2012": (2011, "_G"), "2013-2014": (2013, "_H"), "2015-2016": (2015, "_I"),
    "2017-2018": (2017, "_J"), "2017-2020": (2017, "P_"),  # 2017-March 2020 pre-pandemic files
}
_CORE = ["ALB_CR", "ALQ", "BIOPRO", "BMX", "BPQ", "CBC", "CDQ", "DBQ", "DEMO", "DIQ", "DPQ", "DR1TOT",
         "DR2TOT", "GHB", "GLU", "HDL", "HIQ", "HSQ", "HUQ", "KIQ_U", "MCQ", "OCQ", "PAQ", "RHQ", "RXQ_RX",
         "SLQ", "SMQ", "SMQFAM", "SMQRTU", "TCHOL", "TRIGLY", "WHQ"]
_SURVEY = ["BPX", "DUQ", "HOQ", "PFQ"]  # in every 2-year cycle, not in the pre-pandemic release
FILES = {
    "2005-2006": _CORE + _SURVEY + ["AGQ", "COT", "PAQIAF", "PFC"],
    "2007-2008": _CORE + _SURVEY + ["COTNAL", "INQ", "OGTT", "PFC"],
    "2009-2010": _CORE + _SURVEY + ["COTNAL", "INQ", "OGTT", "PFC"],
    "2011-2012": _CORE + _SURVEY + ["CKQ", "COTNAL", "INQ", "OGTT", "PAXDAY", "PAXHD", "PFC"],
    "2013-2014": _CORE + _SURVEY + ["CKQ", "COT", "INQ", "OGTT", "PAXDAY", "PAXHD", "PFAS", "SSPFAS"],
    "2015-2016": _CORE + _SURVEY + ["COT", "INQ", "OGTT", "PFAS"],
    "2017-2018": _CORE + _SURVEY + ["COT", "INQ", "PFAS"],
    "2017-2020": _CORE + ["BPXO", "COT", "INQ", "PFAS"],
}


def file_name(cycle, comp):
    tag = CYCLES[cycle][1]
    return f"P_{comp}" if tag == "P_" else f"{comp}{tag}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, retries=4):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            with open(tmp, "rb") as f:  # CDC answers some missing files with an HTML page and 200
                if not f.read(80).startswith(b"HEADER RECORD"):
                    tmp.unlink()
                    return "missing", "not an XPT file"
            tmp.rename(dest)
            return "downloaded", None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing", "HTTP 404"
            err = f"HTTP {e.code}"
        except Exception as e:  # network hiccup: retry
            err = repr(e)
        time.sleep(5 * attempt)
    tmp.unlink(missing_ok=True)
    return "failed", err


def main():
    jobs = [(DATA / cycle / f"{file_name(cycle, comp)}.xpt", BASE.format(year=CYCLES[cycle][0], name=file_name(cycle, comp)))
            for cycle, comps in FILES.items() for comp in comps]
    jobs.append((DRUG_FILE, DRUG_URL))
    records = []
    for i, (dest, url) in enumerate(jobs, 1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        rec = {"file": str(dest.relative_to(DATA)), "url": url}
        if dest.exists() or dest.with_suffix(".XPT").exists():
            rec["status"] = "present"
        else:
            rec["status"], rec["error"] = fetch(url, dest)
            if rec["status"] == "downloaded":
                rec["sha256"] = sha256(dest)
        print(f"[{i}/{len(jobs)}] {rec['status']:10} {rec['file']}", flush=True)
        records.append(rec)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (DATA / f"manifest_{ts}.json").write_text(json.dumps({"created_utc": ts, "records": records}, indent=1))
    bad = [r for r in records if r["status"] in ("missing", "failed")]
    print(f"done: {len(records)} files, {len(bad)} not downloaded")
    for r in bad:
        print("  ", r["file"], r["error"])
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
