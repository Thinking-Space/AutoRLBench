"""Public deterministic RGB beacon task, reward prompts and experiment logging."""

import hashlib
import json
import random

from PIL import Image, ImageDraw

PROMPTS = ("Is the illuminated beacon on the left?", "Is the illuminated beacon on the right?")
BATCH = 8
GROUP = 8
SEED = 87021


def beacon(scene_id):
    rng = random.Random(scene_id)
    side = rng.randrange(2)
    image = Image.new("RGB", (224, 224), tuple(rng.randint(15, 45) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(24):
        x, y = rng.randrange(224), rng.randrange(224)
        gray = rng.randrange(40, 85)
        draw.rectangle((x, y, x + rng.randrange(2, 15), y + rng.randrange(2, 15)), fill=(gray,) * 3)
    x = rng.randint(27, 82) + 112 * side
    y = rng.randint(41, 182)
    r = rng.randint(14, 24)
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(250, rng.randint(150, 235), 48), outline=(255, 251, 235), width=3)
    return image, side

def policy_digest(policy):
    h = hashlib.sha256()
    for p in policy.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def report(stream, **data):
    print(json.dumps(data, separators=(",", ":"), allow_nan=False), file=stream, flush=True)
