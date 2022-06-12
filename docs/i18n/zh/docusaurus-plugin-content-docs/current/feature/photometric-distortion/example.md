# 使用示例

简单的可执行调用示例：

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

可以通过 `fireball` (`pip install fireball`) 直接调用以上示例：

```bash
fib vkit_sln.vkit_doc_helper.demo_pho:run \
    --image_file="REQUIRED" \
    --output_file="REQUIRED"
```

以下是示例输入与输出：

<div align="center">
    <img alt="Lenna.png" src="https://i.loli.net/2021/11/25/HFaygJjhuI2OxU1.png" />
	<img alt="demo_output.png" src="https://i.loli.net/2021/11/28/LAvGD7lrkqpa2co.png" />
</div>

下面是光度畸变的具体实现
