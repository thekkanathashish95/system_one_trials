"""Template 4 - BOX: locate something in an image (vision).

Use box to find a region: a signature on a form, a product on a shelf, a defect on a part.
The answer is {top, bottom, left, right} as fractions (0..1) of image height/width, plus
confidence. Images are sent as `assets`; reference the asset key in `state`.

Requires a vision-capable Jev variant (set JEV_VISION_MODEL). Box is the newest question
type, so availability on OpenRouter may lag the TypeSafe API.

    python templates/04_box_locate.py [path/to/image.png] [--dry-run]
"""

import sys
import os
from pathlib import Path

import _common  # noqa: F401

from jev import JevClient, box, image_asset, noul

DEFAULT_IMAGE = _common.ROOT / "assets" / "red_square.png"

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    image = Path(args[0]) if args else DEFAULT_IMAGE
    client = JevClient(model=os.getenv("JEV_VISION_MODEL"))

    d = client.decide(
        state="photo",  # the asset key below; the API swaps it for the image
        assets={"photo": image_asset(image)},
        questions={
            "target": box("The red square"),
            "has_text": noul("Does the image contain any readable text?"),
        },
    )
    if d.answers:
        b = d.box("target")
        print(f"box: top={b['top']:.3f} bottom={b['bottom']:.3f} left={b['left']:.3f} right={b['right']:.3f}"
              f"  confidence={d.confidence('target')}")
        print(f"has_text: {_common.pct(d.yes('has_text'))}")
        # The bundled sample image has the square at top=0.25 bottom=0.625 left=0.375 right=0.75
