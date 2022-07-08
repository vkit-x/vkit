from typing import Union, Tuple, Optional

import numpy as np

from .type import Shapable


def clip_val(val: int, size: int):
    return int(np.clip(val, 0, size - 1))


def resize_val(val: int, size: int, resized_size: int):
    resized_val = round(resized_size * val / size)
    return clip_val(resized_val, resized_size)


def extract_shape_from_shapable_or_shape(shapable_or_shape: Union[Shapable, Tuple[int, int]],):
    if isinstance(shapable_or_shape, Shapable):
        shapable = shapable_or_shape
        height, width = shapable.shape
    else:
        height, width = shapable_or_shape
    return height, width


def generate_resized_shape(
    height: int,
    width: int,
    resized_height: Optional[int] = None,
    resized_width: Optional[int] = None,
):
    if not resized_height and not resized_width:
        raise RuntimeError('Missing resized_height or resized_width.')

    if resized_height is None:
        assert resized_width
        resized_height = round(resized_width * height / width)

    if resized_width is None:
        assert resized_height
        resized_width = round(resized_height * width / height)

    return resized_height, resized_width


def generate_shape_and_resized_shape(
    shapable_or_shape: Union[Shapable, Tuple[int, int]],
    resized_height: Optional[int] = None,
    resized_width: Optional[int] = None,
):
    height, width = extract_shape_from_shapable_or_shape(shapable_or_shape)
    resized_height, resized_width = generate_resized_shape(
        height=height,
        width=width,
        resized_height=resized_height,
        resized_width=resized_width,
    )
    return (
        height,
        width,
        resized_height,
        resized_width,
    )


def expand_np_mask(mat: np.ndarray, np_mask: np.ndarray):
    if mat.ndim == 2:
        # Do nothing.
        pass

    elif mat.ndim == 3:
        num_channels = mat.shape[2]
        np_mask = np.repeat(np.expand_dims(np_mask, axis=-1), num_channels, axis=-1)

    else:
        raise NotImplementedError()

    return np_mask


def prep_value(
    mat: np.ndarray,
    value: Union[np.ndarray, Tuple[float, ...], float],
):
    if not isinstance(value, np.ndarray):
        if mat.ndim == 3:
            num_channels = mat.shape[2]
            if isinstance(value, tuple) and len(value) != num_channels:
                raise RuntimeError('value is tuple but len(value) != num_channels.')

        value = np.full_like(mat, value)

    else:
        if mat.shape != value.shape:
            raise RuntimeError('value is np.ndarray but shape is not matched.')

        if value.dtype != mat.dtype:
            value = value.astype(mat.dtype)

    return value


def fill_np_array(
    mat: np.ndarray,
    value: Union[np.ndarray, Tuple[float, ...], float],
    np_mask: Optional[np.ndarray] = None,
    alpha: Union[np.ndarray, float] = 1.0,
    keep_max_value: bool = False,
    keep_min_value: bool = False,
):
    mat_origin = mat
    np_value = prep_value(mat, value)

    if np_mask is not None:
        # NOTE: Boolean indexing makes a copy.
        mat = mat[np_mask]
        np_value = np_value[np_mask]
        if isinstance(alpha, np.ndarray):
            alpha = alpha[np_mask]

    if isinstance(alpha, float):
        if alpha < 0.0 or alpha > 1.0:
            raise RuntimeError(f'alpha={alpha} is invalid.')

        elif alpha == 0.0:
            return

        elif alpha == 1.0:
            if np_mask is None and not (keep_max_value or keep_min_value):
                np.copyto(mat_origin, np_value)

            else:
                if keep_max_value or keep_min_value:
                    assert not (keep_max_value and keep_min_value)
                    if keep_max_value:
                        np_to_value_mask = (mat < np_value)
                    else:
                        np_to_value_mask = (mat > np_value)
                    np.putmask(mat, np_to_value_mask, np_value)
                else:
                    mat = np_value

                if np_mask is not None:
                    mat_origin[np_mask] = mat

        else:
            # 0 < alpha < 1.
            np_value_weight = np.full(
                (mat.shape[0], mat.shape[1]),
                alpha,
                dtype=np.float32,
            )
            if np_value_weight.shape != mat.shape:
                assert np_value_weight.ndim + 1 == mat.ndim
                np_value_weight = np.expand_dims(np_value_weight, -1)

            np_mat_weight = 1 - np_value_weight
            np_weighted_sum = (
                np_mat_weight * mat.astype(np.float32)
                + np_value_weight * np_value.astype(np.float32)
            )
            np_weighted_sum = np_weighted_sum.astype(mat.dtype)

            if np_mask is not None:
                mat_origin[np_mask] = np_weighted_sum
            else:
                np.copyto(mat_origin, np_weighted_sum)

    else:
        np_value_weight = alpha
        if np_value_weight.shape != mat.shape:
            assert np_value_weight.ndim + 1 == mat.ndim
            np_value_weight = np.expand_dims(np_value_weight, -1)

        np_mat_weight = 1 - np_value_weight
        np_weighted_sum = (
            np_mat_weight * mat.astype(np.float32) + np_value_weight * np_value.astype(np.float32)
        )
        np_weighted_sum = np_weighted_sum.astype(mat.dtype)

        if np_mask is not None:
            mat_origin[np_mask] = np_weighted_sum
        else:
            np.copyto(mat_origin, np_weighted_sum)
