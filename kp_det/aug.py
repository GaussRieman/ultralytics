
import numpy as np
from ultralytics.data.augment import RandomPerspective

transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
result = transform(labels)
transformed_image = result["img"]
transformed_instances = result["instances"]