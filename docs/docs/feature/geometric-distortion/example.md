# Usage

The following example demonstrates how to distort an image by using `camera_cubic_curve` operation：

```python
from vkit.image.type import Image
from vkit.augmentation.geometric_distortion import (
    CameraModelConfig,
    CameraCubicCurveConfig,
    camera_cubic_curve,
)


def run(image_file, output_file):
    image = Image.from_file(image_file)

    config = CameraCubicCurveConfig(
        curve_alpha=60,
        curve_beta=-60,
        curve_direction=0,
        curve_scale=1.0,
        camera_model_config=CameraModelConfig(
            rotation_unit_vec=[1.0, 0.0, 0.0],
            rotation_theta=30,
        ),
        grid_size=10,
    )
    result = camera_cubic_curve.distort(config, image)

    result.image.to_file(output_file)

```

Above script could be executed by `fireball` (`pip install fireball` if you haven't install) directly:

```bash
fib vkit_sln.vkit_doc_helper.demo_geo:run \
    --image_file="REQUIRED" \
    --output_file="REQUIRED"
```

Input and output example：

<div align="center">
    <img alt="Lenna.png" src="/geo/Lenna.png" />
	<img alt="demo_output.png" src="/geo/output.png" />
</div>
