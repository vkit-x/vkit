# Usage

The following example demonstrates how to distort an image by using `mean_shift` operation：

```python
from vkit.image.type import Image
from vkit.augmentation.photometric_distortion import (
    MeanShiftConfig,
    mean_shift,
)


def run(image_file, output_file):
    image = Image.from_file(image_file)

    config = MeanShiftConfig(delta=100)
    new_image = mean_shift.distort_image(config, image)

    new_image.to_file(output_file)
```

Above script could be executed by `fireball` (`pip install fireball` if you haven't install) directly:

```bash
fib vkit_sln.vkit_doc_helper.demo_pho:run \
    --image_file="REQUIRED" \
    --output_file="REQUIRED"
```

Input and output example：

<div align="center">
    <img alt="Lenna.png" src="/pho/Lenna.png" />
	<img alt="demo_output.png" src="/pho/output.png" />
</div>
