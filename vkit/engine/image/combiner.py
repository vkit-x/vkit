# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import Sequence, List, Dict, Optional
import bisect
import heapq

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv
import iolite as io

from vkit.utility import rng_choice, read_json_file
from vkit.element import Image, ImageMode, Mask
from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import ImageEngineRunConfig


@attrs.define(frozen=True)
class ImageMeta:
    image_file: str
    grayscale_mean: float
    grayscale_std: float


class FolderTree:
    IMAGE = 'image'
    METAS_JSON = 'metas.json'


def load_image_metas_from_folder(folder: str):
    in_fd = io.folder(folder, expandvars=True, exists=True)
    image_fd = io.folder(
        in_fd / FolderTree.IMAGE,
        exists=True,
    )
    metas_json = io.file(
        in_fd / FolderTree.METAS_JSON,
        exists=True,
    )

    image_metas: List[ImageMeta] = []
    for meta in read_json_file(metas_json):
        image_file = io.file(image_fd / meta['image_file'], exists=True)
        image_metas.append(
            ImageMeta(
                image_file=str(image_file),
                grayscale_mean=meta['grayscale_mean'],
                grayscale_std=meta['grayscale_std'],
            )
        )

    return image_metas


@attrs.define
class ImageCombinerEngineInitConfig:
    image_meta_folder: str
    target_image_mode: ImageMode = ImageMode.RGB
    enable_cache: bool = False
    sigma: float = 3.0
    init_segment_width_min_ratio: float = 0.25
    gaussian_blur_kernel_size = 5


@attrs.define(order=True)
class PrioritizedSegment:
    y: int = attrs.field(order=True)
    left: int = attrs.field(order=False)
    right: int = attrs.field(order=False)


class ImageCombinerEngine(
    Engine[
        ImageCombinerEngineInitConfig,
        NoneTypeEngineInitResource,
        ImageEngineRunConfig,
        Image,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'combiner'

    def __init__(
        self,
        init_config: ImageCombinerEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        self.image_metas = load_image_metas_from_folder(init_config.image_meta_folder)
        self.image_metas = sorted(
            self.image_metas,
            key=lambda meta: meta.grayscale_mean,
        )
        self.image_metas_grayscale_means = [
            image_meta.grayscale_mean for image_meta in self.image_metas
        ]
        self.enable_cache = init_config.enable_cache
        self.image_file_to_cache_image: Dict[str, Image] = {}

    def sample_image_metas_based_on_random_anchor(
        self,
        run_config: ImageEngineRunConfig,
        rng: RandomGenerator,
    ):
        # Get candidates based on anchor.
        anchor_image_meta = rng_choice(rng, self.image_metas)
        grayscale_std = anchor_image_meta.grayscale_std
        grayscale_mean = anchor_image_meta.grayscale_mean

        grayscale_begin = round(grayscale_mean - self.init_config.sigma * grayscale_std)
        grayscale_end = round(grayscale_mean + self.init_config.sigma * grayscale_std)

        index_begin = bisect.bisect_left(self.image_metas_grayscale_means, x=grayscale_begin)
        index_end = bisect.bisect_right(self.image_metas_grayscale_means, x=grayscale_end)
        image_metas = self.image_metas[index_begin:index_end]

        assert image_metas
        return image_metas

    @classmethod
    def fill_np_edge_mask(
        cls,
        np_edge_mask: np.ndarray,
        height: int,
        width: int,
        gaussian_blur_half_kernel_size: int,
        up: int,
        down: int,
        left: int,
        right: int,
    ):
        # Fill up.
        up_min = max(0, up - gaussian_blur_half_kernel_size)
        up_max = min(height - 1, up + gaussian_blur_half_kernel_size)
        np_edge_mask[up_min:up_max + 1, left:right + 1] = 1

        # Fill down.
        down_min = max(0, down - gaussian_blur_half_kernel_size)
        down_max = min(height - 1, down + gaussian_blur_half_kernel_size)
        np_edge_mask[down_min:down_max + 1, left:right + 1] = 1

        # Fill left.
        left_min = max(0, left - gaussian_blur_half_kernel_size)
        left_max = min(width - 1, left + gaussian_blur_half_kernel_size)
        np_edge_mask[up:down + 1, left_min:left_max + 1] = 1

        # Fill right.
        right_min = max(0, right - gaussian_blur_half_kernel_size)
        right_max = min(width - 1, right + gaussian_blur_half_kernel_size)
        np_edge_mask[up:down + 1, right_min:right_max + 1] = 1

    def synthesize_image(
        self,
        run_config: ImageEngineRunConfig,
        image_metas: Sequence[ImageMeta],
        rng: RandomGenerator,
    ):
        height = run_config.height
        width = run_config.width

        mat = np.zeros((height, width, 3), dtype=np.uint8)
        edge_mask = Mask.from_shape((height, width))
        gaussian_blur_half_kernel_size = self.init_config.gaussian_blur_kernel_size // 2 + 1

        # Initialize segments.
        priority_queue: List[PrioritizedSegment] = []
        segment_width_min = int(
            np.clip(
                round(self.init_config.init_segment_width_min_ratio * width),
                1,
                width - 1,
            )
        )
        left = 0
        while left + segment_width_min - 1 < width:
            right = rng.integers(
                left + segment_width_min - 1,
                width,
            )
            if right + 1 - left < segment_width_min or width - right - 1 < segment_width_min:
                break
            priority_queue.append(PrioritizedSegment(
                y=0,
                left=left,
                right=right,
            ))
            left = right + 1
        if left < width:
            priority_queue.append(PrioritizedSegment(
                y=0,
                left=left,
                right=width - 1,
            ))

        while priority_queue:
            # Pop a segment
            cur_segment = heapq.heappop(priority_queue)

            # Deal with connection.
            segments: List[PrioritizedSegment] = []
            while priority_queue and priority_queue[0].y == cur_segment.y:
                segments.append(heapq.heappop(priority_queue))

            if segments:
                segments.append(cur_segment)
                segments = sorted(segments, key=lambda segment: segment.left)
                cur_segment_idx = -1
                for segment_idx, segment in enumerate(segments):
                    if segment.left == cur_segment.left and segment.right == cur_segment.right:
                        cur_segment_idx = segment_idx
                        break
                assert cur_segment_idx >= 0

                begin = cur_segment_idx
                while begin > 0 and segments[begin - 1].right + 1 == segments[begin].left:
                    begin -= 1
                end = cur_segment_idx
                while end + 1 < len(segments) and segments[end].right + 1 == segments[end + 1].left:
                    end += 1

                if begin < end:
                    # Update the current segment.
                    cur_segment.left = segments[begin].left
                    cur_segment.right = segments[end].right

                # Push back.
                for segment in segments[:begin]:
                    heapq.heappush(priority_queue, segment)
                for segment in segments[end + 1:]:
                    heapq.heappush(priority_queue, segment)

            # Load image.
            image_meta = rng_choice(rng, image_metas)
            if self.enable_cache and image_meta.image_file in self.image_file_to_cache_image:
                segment_image = self.image_file_to_cache_image[image_meta.image_file]
            else:
                segment_image = Image.from_file(image_meta.image_file).to_target_mode_image(
                    self.init_config.target_image_mode
                )
                if self.enable_cache:
                    self.image_file_to_cache_image[image_meta.image_file] = segment_image

            # Fill image and edge mask.
            up = cur_segment.y
            down = min(height - 1, up + segment_image.height - 1)
            left = cur_segment.left
            right = min(cur_segment.right, left + segment_image.width - 1)
            mat[up:down + 1, left:right + 1] = \
                segment_image.mat[:down + 1 - up, :right + 1 - left]

            with edge_mask.writable_context:
                self.fill_np_edge_mask(
                    np_edge_mask=edge_mask.mat,
                    height=height,
                    width=width,
                    gaussian_blur_half_kernel_size=gaussian_blur_half_kernel_size,
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                )

            # Update segments.
            if right == cur_segment.right:
                # Reach the current right end.
                cur_segment.y = down + 1
                if cur_segment.y < height:
                    heapq.heappush(priority_queue, cur_segment)
            else:
                # Not reaching the right end.
                assert right < cur_segment.right
                new_segment = PrioritizedSegment(
                    y=down + 1,
                    left=left,
                    right=right,
                )
                if new_segment.y < height:
                    heapq.heappush(priority_queue, new_segment)

                cur_segment.left = right + 1
                heapq.heappush(priority_queue, cur_segment)

        # Apply gaussian blur.
        gaussian_blur_sigma = gaussian_blur_half_kernel_size / 3
        gaussian_blur_ksize = (self.init_config.gaussian_blur_kernel_size,) * 2
        edge_mask.fill_np_array(
            mat,
            cv.GaussianBlur(mat, gaussian_blur_ksize, gaussian_blur_sigma),
        )

        return Image(mat=mat)

    def run(self, run_config: ImageEngineRunConfig, rng: RandomGenerator) -> Image:
        assert not run_config.disable_resizing
        image_metas = self.sample_image_metas_based_on_random_anchor(run_config, rng)
        return self.synthesize_image(run_config, image_metas, rng)


image_combiner_engine_executor_factory = EngineExecutorFactory(ImageCombinerEngine)
