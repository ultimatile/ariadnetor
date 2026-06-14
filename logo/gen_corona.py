"""Generate the ariadnetor Corona Borealis mark (nodes + lines only).

The mark is the constellation Corona Borealis drawn as a tensor-network-style
graph: circular nodes joined by straight edges. Node positions are the *real*
sky coordinates of the seven crown stars (epoch J2000; source noted on STARS
below), so the arc is naturally irregular (not a true circle) and opens downward
like a shallow cup.

The conventional stick-figure order theta-beta-alpha-gamma-delta-epsilon-iota
puts Alphecca (alpha CrB, the crown's jewel and brightest star) third in the
chain, so the 3rd node is the single op-colored (red) accent.

Usage:
    python gen_corona.py              # square icon SVG
    python gen_corona.py --wordmark   # icon + "ariadnetor" lockup
    python gen_corona.py --png        # also export PNG via rsvg-convert (librsvg)
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

# ── Palette ─────────────────────────────────────────────────
COLOR_BG = "#1a1a2e"
COLOR_EDGE = "#2d4a6f"   # constellation line
COLOR_NODE = "#4a90d9"   # regular star node
COLOR_O = "#e06040"      # operator highlight = Alphecca
COLOR_TEXT = "#ffffff"

# ── Wordmark ────────────────────────────────────────────────
WORD = "ariadnetor"
# Fonts explored before settling on Lexend (legibility-led, no heavy "pressure"):
#   "'Iceland'"                 -- the previous wordmark; condensed, low legibility
#   "'Azeret Mono','monospace'" -- the only other font actually trialled
#   "'Space Grotesk'"           -- nice but felt sparse at this word
#   "'Atkinson Hyperlegible'"   -- max legibility, but softer/warmer than wanted
FONT_FAMILY = "'Lexend'"
FONT_WEIGHT = 500     # Lexend Medium: legible without the Bold "pressure"
LETTER_SPACING = 2

# ── Star data ───────────────────────────────────────────────
# Seven crown stars as (label: (RA[hours], Dec[deg])), epoch J2000.
#
# Source: English Wikipedia "List of stars in Corona Borealis", which tabulates
# RA/Dec (J2000, arcsec precision; the page cites SIMBAD) for every crown star
# in one place: https://en.wikipedia.org/wiki/List_of_stars_in_Corona_Borealis
# Spot-checked: Alphecca lists RA 15h34m41.19s = 15.5781h, Dec +26d42'53.7" =
# +26.7149deg, matching the value below. Stellar positions are facts (not
# copyrightable); this note records provenance per data-attribution practice,
# not as a license obligation.
#
# Chain order below is the drawn polyline; Alphecca is index 2 (the 3rd node),
# the single red accent.
STARS = {
    "theta":   (15.549, 31.359),  # mag 4.1
    "beta":    (15.464, 29.106),  # mag 3.7  Nusakan
    "alpha":   (15.578, 26.715),  # mag 2.2  Alphecca
    "gamma":   (15.712, 26.296),  # mag 3.8
    "delta":   (15.827, 26.068),  # mag 4.6
    "epsilon": (15.960, 26.878),  # mag 4.1
    "iota":    (16.024, 29.851),  # mag 4.9
}
CHAIN = ["theta", "beta", "alpha", "gamma", "delta", "epsilon", "iota"]
ALPHECCA = "alpha"

# ── Geometry ────────────────────────────────────────────────
ICON = 200            # square icon side in px
PAD = 34              # padding from icon edge to star bounding box
NODE_R = 6            # regular node radius
ALPHECCA_R = 10       # Alphecca accent (the orthogonality-center diamond)
EDGE_WIDTH = 2.5      # constellation line; bumped from 2.2 for favicon legibility

FONT_SIZE = 104       # wordmark size in the lockup
LOCKUP_ICON_SCALE = 1.45  # icon enlarged in the lockup to balance the wordmark
TEXT_GAP = 28         # gap between icon and wordmark
MARGIN_Y = 10


def raw_points():
    """Project (RA, Dec) to a y-up plane (larger y = higher in the sky).

    cos(Dec) flattens RA to true angular spacing, and RA is negated because a sky
    chart is the view looking *up* from Earth, where east (increasing RA) is to
    the LEFT. The un-negated form is the mirror image (the view from outside the
    celestial sphere) -- which reads backwards vs the standard star chart.
    """
    dec0 = math.radians(sum(d for _, d in STARS.values()) / len(STARS))
    return {name: (-ra * 15.0 * math.cos(dec0), dec)
            for name, (ra, dec) in STARS.items()}


def projected_points():
    """Fit the projected coords into the padded square (uniform scale, centered).
    The Dec/up convention lands theta/iota high on the ends and Alphecca low at
    the cup's bottom -- the downward-opening crown."""
    raw = raw_points()
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny)  # uniform scale keeps the shape true
    box = ICON - 2 * PAD
    scale = box / span

    # center the (possibly non-square) star box inside the padded area
    offx = PAD + (box - (maxx - minx) * scale) / 2
    offy = PAD + (box - (maxy - miny) * scale) / 2

    pts = {}
    for name, (x, y) in raw.items():
        px = offx + (x - minx) * scale
        py = offy + (maxy - y) * scale  # flip: up -> SVG y down
        pts[name] = (px, py)
    return pts


def _alphecca_tangent(placed):
    """Angle (radians) of the chain passing through Alphecca, in placed/display
    coords. Used to orient the orthogonality-center diamond so its diagonal lies
    along the chain -- i.e. the two incident edges enter at opposite vertices
    (ports), as tensor-network diagrams draw bonds into a node's corners."""
    idx = CHAIN.index(ALPHECCA)
    ax, ay = placed[CHAIN[idx]]
    px, py = placed[CHAIN[idx - 1]]   # incoming neighbour (beta)
    nx, ny = placed[CHAIN[idx + 1]]   # outgoing neighbour (gamma)

    def unit(dx, dy):
        h = math.hypot(dx, dy) or 1.0
        return dx / h, dy / h

    ix, iy = unit(ax - px, ay - py)   # beta -> Alphecca
    ox_, oy_ = unit(nx - ax, ny - ay)  # Alphecca -> gamma
    return math.atan2(iy + oy_, ix + ox_)


def render_mark(pts, ox=0.0, oy=0.0, scale=1.0,
                alphecca_shape="diamond", alphecca_r=ALPHECCA_R,
                alphecca_angle=None, edge_width=EDGE_WIDTH):
    """Emit the constellation graph (edges then nodes). Coords are scaled about
    the origin then offset by (ox, oy); stroke width and node radii scale too, so
    the whole mark grows uniformly (used to balance the icon against the
    wordmark in a lockup).

    Alphecca (the red accent) renders as a diamond: it evokes a tensor-network
    orthogonality center -- the one distinguished MPS site -- which lines up with
    Alphecca being the crown's lone bright jewel. alphecca_angle (degrees) sets
    the diamond's diagonal direction; None aligns it with the chain tangent so
    the two incident edges land on opposite vertices (the bond ports). A circle
    is still available via alphecca_shape="circle".
    """
    s = []
    a = s.append

    def place(x, y):
        return x * scale + ox, y * scale + oy

    placed = {name: place(x, y) for name, (x, y) in pts.items()}

    # constellation lines
    a(f'  <g stroke="{COLOR_EDGE}" stroke-width="{edge_width * scale:.2f}"'
      f' fill="none" stroke-linecap="round" stroke-linejoin="round">')
    d = "M " + " L ".join(f"{placed[n][0]:.2f},{placed[n][1]:.2f}" for n in CHAIN)
    a(f'    <path d="{d}"/>')
    a("  </g>")

    # regular star nodes
    for name, (px, py) in placed.items():
        if name == ALPHECCA:
            continue
        a(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="{NODE_R * scale:.2f}"'
          f' fill="{COLOR_NODE}"/>')

    # Alphecca (drawn last so its accent sits on top)
    ax, ay = placed[ALPHECCA]
    r = alphecca_r * scale
    if alphecca_shape == "diamond":
        phi = (_alphecca_tangent(placed) if alphecca_angle is None
               else math.radians(alphecca_angle))
        c, sn = math.cos(phi), math.sin(phi)
        # two vertices along the chain tangent (phi), two on the perpendicular
        verts = [(ax + r * c, ay + r * sn), (ax - r * sn, ay + r * c),
                 (ax - r * c, ay - r * sn), (ax + r * sn, ay - r * c)]
        pgon = " ".join(f"{vx:.2f},{vy:.2f}" for vx, vy in verts)
        a(f'  <polygon points="{pgon}" fill="{COLOR_O}" stroke-linejoin="round"/>')
    else:
        a(f'  <circle cx="{ax:.2f}" cy="{ay:.2f}" r="{r:.2f}" fill="{COLOR_O}"/>')
    return s


def generate_icon(pts, background=COLOR_BG) -> str:
    """Square icon. background=None leaves it transparent; the node/edge/diamond
    colours read on both light and dark canvases, so one transparent icon serves
    both themes (no text to invert)."""
    s = ['<?xml version="1.0" encoding="UTF-8"?>']
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON} {ICON}"'
             f' width="{ICON}" height="{ICON}">')
    if background:
        s.append(f'  <rect width="{ICON}" height="{ICON}" fill="{background}"/>')
    s += render_mark(pts)
    s.append("</svg>")
    return "\n".join(s)


def generate_lockup(pts, background=COLOR_BG, text_color=COLOR_TEXT) -> str:
    # The icon is enlarged (LOCKUP_ICON_SCALE) so the constellation holds its own
    # next to the wordmark; the SVG height is the taller of icon vs text.
    # background=None + a theme-appropriate text_color yields the light/dark
    # variants paired via <picture> (only the wordmark colour need invert).
    icon_h = ICON * LOCKUP_ICON_SCALE
    avg = FONT_SIZE * 0.55  # Lexend lowercase avg advance
    text_w = len(WORD) * avg + (len(WORD) - 1) * LETTER_SPACING
    svg_h = max(icon_h, FONT_SIZE * 1.25)
    svg_w = icon_h + TEXT_GAP + text_w + MARGIN_Y

    s = ['<?xml version="1.0" encoding="UTF-8"?>']
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}"'
             f' width="{svg_w:.0f}" height="{svg_h:.0f}">')
    if background:
        s.append(f'  <rect width="{svg_w:.0f}" height="{svg_h:.0f}" fill="{background}"/>')
    # vertically center the icon in the canvas
    s += render_mark(pts, ox=0, oy=(svg_h - icon_h) / 2, scale=LOCKUP_ICON_SCALE)

    tx = icon_h + TEXT_GAP
    ty = svg_h / 2
    style = f"letter-spacing: {LETTER_SPACING}px;"
    s.append(f'  <text x="{tx:.0f}" y="{ty:.0f}" font-family="{FONT_FAMILY}"'
             f' font-size="{FONT_SIZE}" font-weight="{FONT_WEIGHT}"'
             f' style="{style}" text-anchor="start" dominant-baseline="central"'
             f' fill="{text_color}">{WORD}</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_to_png(svg_path: Path, png_path: Path, scale: int = 2):
    """Rasterize SVG to PNG with rsvg-convert (librsvg) at `scale`x the SVG's
    pixel size. It is a PATH command (no hard-coded app path), and resolves fonts
    through fontconfig -- so the lockup's Lexend wordmark bakes in correctly as
    long as Lexend is installed. The mark stays transparent where the SVG has no
    background rect (rsvg-convert defaults to a transparent canvas)."""
    if shutil.which("rsvg-convert") is None:
        raise SystemExit(
            "rsvg-convert not found; install librsvg (e.g. `brew install librsvg`)"
        )
    subprocess.run(
        ["rsvg-convert", "--zoom", str(scale),
         "--output", str(png_path), str(svg_path)],
        check=True,
    )
    print(f"  -> {png_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate ariadnetor Corona Borealis mark")
    ap.add_argument("--wordmark", action="store_true",
                    help="Emit the icon + 'ariadnetor' lockup as well")
    ap.add_argument("--png", action="store_true", help="Also export PNG via rsvg-convert")
    ap.add_argument("--scale", type=int, default=2, help="PNG scale factor (default: 2)")
    args = ap.parse_args()

    here = Path(__file__).parent
    pts = projected_points()

    # Each entry: filename stem -> SVG string. The navy-background versions are
    # the standalone assets; the _transparent / _dark / _light variants are for
    # theme-aware <picture> embedding (transparent icon serves both themes; the
    # lockup needs a dark/light pair because only the wordmark colour inverts).
    assets = {
        "corona_icon": generate_icon(pts),
        "corona_icon_transparent": generate_icon(pts, background=None),
    }
    if args.wordmark:
        assets["corona_lockup"] = generate_lockup(pts)
        assets["corona_lockup_dark"] = generate_lockup(pts, background=None, text_color=COLOR_TEXT)
        assets["corona_lockup_light"] = generate_lockup(pts, background=None, text_color=COLOR_BG)

    for stem, svg in assets.items():
        svg_path = here / f"{stem}.svg"
        svg_path.write_text(svg)
        print(f"wrote {svg_path.name}")
        if args.png:
            svg_to_png(svg_path, svg_path.with_suffix(".png"), scale=args.scale)


if __name__ == "__main__":
    main()
