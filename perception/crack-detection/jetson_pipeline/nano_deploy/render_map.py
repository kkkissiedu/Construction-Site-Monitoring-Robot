"""render_map.py - turn a survey detection log into an interactive defect map.

Host-side (laptop) companion to survey.py. Reads ``detections.jsonl`` plus the
thumbnails, places each detection at its GPS coordinate, and writes:

* ``defect_map.html`` - a self-contained Leaflet map with severity-coloured
  markers, embedded thumbnail popups, and the survey track polyline.
* ``report.csv`` - a tabular defect report (coords, severity, metrics).
* ``defect_map_preview.png`` - a static preview (no internet needed to view).

Coordinates: detections logged with ``gps_fix: false`` are laid onto a
placeholder track (metric geolocation via a stereo-camera setup is planned;
this is the current state of a larger system). Records that carry a real fix
are placed at their true coordinates and the placeholder step is skipped.

Usage:
    python render_map.py --survey survey_out --out-dir survey_out
"""

from __future__ import annotations

# %% Imports
import argparse
import base64
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %% Config
# Placeholder track origin (Accra, Ghana - fits the GhanaCrack project). Only
# used when detections lack a real GPS fix, pending stereo-camera geolocation.
SIM_ORIGIN_LAT = 5.60370
SIM_ORIGIN_LON = -0.18700
SIM_HEADING_DEG = 78.0          # roughly WSW->ENE along a street
SIM_SPACING_M = 22.0            # metres between successive detections
SIM_CURVE_DEG = 1.4            # small heading drift per step, for a natural path

SEVERITY_COLOUR = {
    "minor": "#2ECC71",
    "moderate": "#F5A623",
    "severe": "#E74C3C",
}
METRES_PER_DEG_LAT = 111320.0


# %%
def load_detections(log_path: Path) -> list[dict[str, Any]]:
    """Load the JSONL detection log.

    Args:
        log_path: Path to ``detections.jsonl``.

    Returns:
        The detection records in file order.

    Raises:
        FileNotFoundError: If the log is missing.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"{log_path} not found - run survey.py first.")
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# %%
def simulate_track(records: list[dict[str, Any]]) -> bool:
    """Assign simulated coordinates to detections lacking a real GPS fix.

    Records are placed along a gently curving polyline starting at the
    configured origin. Records that already carry a real fix are left untouched.

    Args:
        records: Detection records, mutated in place with ``lat``/``lon``.

    Returns:
        ``True`` if any coordinate was simulated (map should say so).
    """
    simulated = False
    lat, lon, heading = SIM_ORIGIN_LAT, SIM_ORIGIN_LON, SIM_HEADING_DEG
    for rec in records:
        if rec.get("gps_fix") and rec.get("lat") is not None:
            continue
        rec["lat"], rec["lon"], rec["simulated"] = lat, lon, True
        simulated = True
        # Advance along the heading by SIM_SPACING_M, drifting slightly.
        d_north = SIM_SPACING_M * math.cos(math.radians(heading))
        d_east = SIM_SPACING_M * math.sin(math.radians(heading))
        lat += d_north / METRES_PER_DEG_LAT
        lon += d_east / (METRES_PER_DEG_LAT * math.cos(math.radians(lat)))
        heading += SIM_CURVE_DEG
    return simulated


# %%
def embed_thumb(thumb_path: Path) -> str:
    """Return a base64 data URI for a thumbnail, or empty string if missing.

    Args:
        thumb_path: Path to the JPEG thumbnail.

    Returns:
        A ``data:image/jpeg;base64,...`` URI, or ``""``.
    """
    if not thumb_path.is_file():
        return ""
    data = base64.b64encode(thumb_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


# %%
def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write the tabular defect report.

    Args:
        records: Detection records with coordinates assigned.
        path: Output CSV path.
    """
    cols = ["id", "timestamp", "source", "severity", "area_pct", "length_px",
            "width_px", "lat", "lon", "gps_fix", "n_detections"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        for rec in records:
            writer.writerow([rec.get(c, "") for c in cols])


# %%
def render_preview(records: list[dict[str, Any]], simulated: bool, path: Path) -> None:
    """Render a static PNG preview of the track and detections.

    Args:
        records: Detection records with coordinates.
        simulated: Whether the track is simulated (shown in the title).
        path: Output PNG path.
    """
    lons = [r["lon"] for r in records]
    lats = [r["lat"] for r in records]
    colours = [SEVERITY_COLOUR.get(r["severity"], "#888888") for r in records]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(lons, lats, "-", color="#555555", linewidth=1.5, alpha=0.7, zorder=1,
            label="survey track")
    ax.scatter(lons, lats, c=colours, s=140, edgecolors="black", linewidths=0.8,
               zorder=2)
    for r in records:
        ax.annotate(str(r["id"]), (r["lon"], r["lat"]), fontsize=7,
                    ha="center", va="center", color="white", zorder=3)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="black", label=s)
               for s, c in SEVERITY_COLOUR.items()]
    ax.legend(handles=handles, title="Severity", loc="best")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    title = "Concrete crack survey - {} detections along track".format(len(records))
    if simulated:
        title += ("\n(placeholder coordinates - metric geolocation via stereo "
                  "camera planned; detections are real model outputs)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# %%
def render_html(
    records: list[dict[str, Any]],
    survey_dir: Path,
    simulated: bool,
    path: Path,
) -> None:
    """Write the self-contained Leaflet defect map.

    Args:
        records: Detection records with coordinates.
        survey_dir: Directory holding the thumbnails.
        simulated: Whether the track is simulated (shown in a banner).
        path: Output HTML path.
    """
    features = []
    track = []
    for rec in records:
        track.append([rec["lat"], rec["lon"]])
        thumb = embed_thumb(survey_dir / rec.get("thumb", ""))
        popup = (
            "<b>Defect #{id}</b><br>Severity: <b>{sev}</b><br>"
            "Area: {area}% &nbsp; Width: ~{w}px<br>{ts}"
        ).format(id=rec["id"], sev=rec["severity"], area=rec["area_pct"],
                 w=rec["width_px"], ts=rec["timestamp"])
        if thumb:
            popup += '<br><img src="{}" style="width:260px;margin-top:6px;">'.format(thumb)
        features.append({
            "lat": rec["lat"], "lon": rec["lon"],
            "colour": SEVERITY_COLOUR.get(rec["severity"], "#888888"),
            "popup": popup,
        })

    counts = {"minor": 0, "moderate": 0, "severe": 0}
    for r in records:
        counts[r["severity"]] = counts.get(r["severity"], 0) + 1
    centre = [sum(f["lat"] for f in features) / len(features),
              sum(f["lon"] for f in features) / len(features)]

    banner = (
        "Placeholder coordinates (metric geolocation via stereo camera planned). "
        "Detections and severity are real model outputs."
        if simulated else "Live GPS survey."
    )

    html = _HTML_TEMPLATE.format(
        centre=json.dumps(centre),
        features=json.dumps(features),
        track=json.dumps(track),
        banner=banner,
        n=len(records), minor=counts["minor"],
        moderate=counts["moderate"], severe=counts["severe"],
    )
    path.write_text(html, encoding="utf-8")


# %%
_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crack Defect Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{{margin:0;font-family:system-ui,Arial,sans-serif}}
 #map{{height:100vh}}
 .banner{{position:absolute;top:10px;left:50px;z-index:1000;background:#fff;padding:8px 12px;
   border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:13px;max-width:60%}}
 .legend{{position:absolute;bottom:20px;right:10px;z-index:1000;background:#fff;padding:8px 12px;
   border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:13px}}
 .dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;border:1px solid #000}}
</style></head><body>
<div id="map"></div>
<div class="banner"><b>GhanaCrack - on-device crack survey</b><br>{banner}<br>
  {n} defects: {minor} minor, {moderate} moderate, {severe} severe</div>
<div class="legend">
  <span class="dot" style="background:#2ECC71"></span>minor<br>
  <span class="dot" style="background:#F5A623"></span>moderate<br>
  <span class="dot" style="background:#E74C3C"></span>severe</div>
<script>
 var map=L.map('map').setView({centre},17);
 L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
   {{maxZoom:19,attribution:'&copy; OpenStreetMap'}}).addTo(map);
 var track={track};
 if(track.length>1) L.polyline(track,{{color:'#333',weight:3,opacity:0.6,dashArray:'6,6'}}).addTo(map);
 var feats={features};
 feats.forEach(function(f){{
   L.circleMarker([f.lat,f.lon],{{radius:9,color:'#000',weight:1,
     fillColor:f.colour,fillOpacity:0.9}}).addTo(map).bindPopup(f.popup);
 }});
</script></body></html>
"""


# %% Main
def main(config: dict[str, Any]) -> int:
    """Build the map, report and preview from a survey log.

    Args:
        config: Runtime configuration.

    Returns:
        Process exit code.
    """
    survey_dir = Path(config["survey"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_detections(survey_dir / "detections.jsonl")
    if not records:
        print("[map] no detections to render")
        return 1
    simulated = simulate_track(records)

    html_path = out_dir / "defect_map.html"
    csv_path = out_dir / "report.csv"
    png_path = out_dir / "defect_map_preview.png"
    render_html(records, survey_dir, simulated, html_path)
    write_csv(records, csv_path)
    render_preview(records, simulated, png_path)

    print("[map] {} defects rendered ({}simulated track)".format(
        len(records), "" if simulated else "real "))
    print("[map] interactive map -> {}".format(html_path))
    print("[map] report         -> {}".format(csv_path))
    print("[map] preview        -> {}".format(png_path))
    return 0


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a survey log into a defect map.")
    parser.add_argument("--survey", default="survey_out",
                        help="directory holding detections.jsonl and thumbs/")
    parser.add_argument("--out-dir", dest="out_dir", default="survey_out")
    args = parser.parse_args()
    raise SystemExit(main(vars(args)))
