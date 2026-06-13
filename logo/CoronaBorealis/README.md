# Corona Borealis mark

![ariadnetor lockup](corona_lockup.png)

The ariadnetor logo: the constellation Corona Borealis as a node-and-edge graph,
its brightest star Alphecca picked out in red.

In myth, Corona Borealis is **Ariadne's Crown** — the constellation already bore
the name. Alphecca's diamond echoes how tensor-network diagrams single out a
distinguished node.

![icon](corona_icon.png)

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

Shares the ariadnetor logo palette; see `../README.md`.
