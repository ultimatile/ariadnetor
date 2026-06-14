# Corona Borealis mark

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="corona_lockup_dark.png">
  <img src="corona_lockup_light.png" alt="ariadnetor" width="520">
</picture>

The ariadnetor logo: the constellation Corona Borealis as a node-and-edge graph,
its brightest star Alphecca picked out in red.

In myth, Corona Borealis is **Ariadne's Crown** — the constellation already bore
the name. Alphecca's diamond echoes how tensor-network diagrams single out a
distinguished node.

<img src="corona_icon_transparent.png" alt="Corona Borealis icon" width="160">

The icon is transparent, so one file works on both light and dark backgrounds;
the lockup ships as a light/dark pair (above) since only the wordmark colour
needs to invert.

## Generate

```
python gen_corona.py            # square icon
python gen_corona.py --wordmark # + "ariadnetor" lockup
python gen_corona.py --png      # also export PNG (headless Chrome)
```

## Star data

The seven nodes sit at the crown stars' real J2000 positions, so the arc is
irregular, not a clean circle. Coordinates come from each star's English
Wikipedia infobox (from SIMBAD / Hipparcos); see `STARS` in `gen_corona.py`.

## Palette

| role              | hex       |
|-------------------|-----------|
| background        | `#1a1a2e` |
| node              | `#4a90d9` |
| edge              | `#2d4a6f` |
| accent (Alphecca) | `#e06040` |
| wordmark          | `#ffffff` |
