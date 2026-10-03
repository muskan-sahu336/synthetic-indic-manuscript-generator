import numpy as np
from PIL import Image
from src.backgrounds import make_paper, make_palm_leaf

for i in range(3):
    rng = np.random.default_rng(i)
    Image.fromarray(make_paper(rng=rng)["image"]).save(f"output/bg_paper_{i}.png")
    Image.fromarray(make_palm_leaf(rng=rng)["image"]).save(f"output/bg_palm_{i}.png")
print("done")