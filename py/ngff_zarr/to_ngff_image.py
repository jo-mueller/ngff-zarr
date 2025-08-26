from collections.abc import MutableMapping
from typing import Hashable, Mapping, Optional, Sequence, Union, List

import dask
from dask.array.core import Array as DaskArray
from numpy.typing import ArrayLike

try:
    from zarr.core import Array as ZarrArray
except ImportError:
    from zarr.core.array import Array as ZarrArray

from .methods._support import _spatial_dims
from .ngff_image import NgffImage
from .v04.zarr_metadata import SupportedDims, Units
from .rfc5 import NgffBaseTransformation, NgffCoordinateSystem, NgffScale, NgffTranslation, NgffAxis, NgffSequence

def to_ngff_image(
    data: Union[ArrayLike, MutableMapping, str, ZarrArray],
    dims: Optional[Sequence[SupportedDims]] = None,
    transformations: Optional[Union[List[NgffBaseTransformation], NgffBaseTransformation]] = None,
    scale: Optional[Union[Mapping[Hashable, float], NgffScale]] = None,
    translation: Optional[Union[Mapping[Hashable, float], NgffTranslation]] = None,
    name: str = "image",
    coordinate_system_name: str = "physical",
    axes_units: Optional[Mapping[str, Units]] = None,
) -> NgffImage:
    """
    Create an image with pixel array and metadata to following the OME-NGFF data model.

    :param data: Multi-dimensional array that provides the image pixel values. It can be a numpy.ndarray
         or another type that behaves like a numpy.ndarray, i.e. an ArrayLike.
         If a ZarrArray, MutableMapping, or str, it will be loaded into Dask lazily
         as a zarr Array.
    :type  data: ArrayLike, ZarrArray, MutableMapping, str

    :param dims: Tuple specifying the data dimensions.
        Values should drawn from: {'t', 'z', 'y', 'x', 'c'} for time, third spatial direction
        second spatial direction, first spatial dimension, and channel or
        component, respectively spatial dimension, and time, respectively.
    :type  dims: sequence of hashable, optional

    :param scale: Pixel spacing for the spatial dims
    :type  scale: dict of floats, optional

    :param translation: Origin or offset of the center of the first pixel.
    :type  translation: dict of floats, optional

    :param name: Name of the resulting image
    :type  name: str, optional

    :param axes_units: Units to associate with the axes. Should be drawn from UDUNITS-2, enumerated at
        https://ngff.openmicroscopy.org/latest/#axes-md
    :type  axes_units: dict of str, optional

    :return: Representation of an image (pixel data + metadata) for a single scale of an NGFF-OME-Zarr multiscale dataset
    :rtype: NgffImage
    """

    if not isinstance(data, DaskArray):
        if isinstance(data, (ZarrArray, str, MutableMapping)):
            data = dask.array.from_zarr(data)
        else:
            data = dask.array.from_array(data)

    # passed transformations should supersede the "old" scale/translation/etc
    if transformations is not None:
        if isinstance(transformations, list):
            transformations = NgffSequence(transformations)

        return NgffImage(
            data=data,
            transformations=transformations,
            name=name,
        )

    ndim = data.ndim
    ax_type_dispatch = {
        'z': {
            'type': 'space',
            'discrete': False
        },
        'y': {
            'type': 'space',
            'discrete': False
        },
        'x': {
            'type': 'space',
            'discrete': False
        },
        'c': {
            'type': 'channel',
            'discrete': True
        },
        't': {
            'type': 'time',
            'discrete': False
        }
    }

    # create default coordinate system only if dims is provided
    _supported_dims = {"c", "x", "y", "z", "t"}
    if dims is not None:
        if not set(dims).issubset(_supported_dims):
            raise ValueError("dims not valid")
        axes = [NgffAxis(name=dim, type=ax_type_dispatch[dim]['type']) for dim in dims]
        output_coordinate_system = NgffCoordinateSystem(name=coordinate_system_name, axes=axes)

    # Set axis units if passed
    if axes_units is not None:
        for ax, unit in axes_units.items():
            output_coordinate_system.set_unit(ax, unit)

    # convert scale and translation to ngff transformation
    transformations = None
    if scale is not None:
        transformations = NgffScale(scale, output_coordinate_system=output_coordinate_system)

    if translation is not None:
        translation = NgffTranslation(translation, output_coordinate_system=output_coordinate_system)
        if transformations is None:
            transformations = translation
        else:
            transformations = NgffSequence([transformations, translation])

    if transformations is None:
        transformations = NgffScale(
            [1.0 for _ in range(ndim)]
        )

    return NgffImage(
        data=data,
        transformations=transformations,
        name=name,
    )
