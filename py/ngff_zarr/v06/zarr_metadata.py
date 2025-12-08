# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Union

from ..v04.zarr_metadata import Axis, Omero, MethodMetadata

@dataclass
class CoordinateSystem:
    name: str
    axes: List[Axis]

@dataclass
class Scale:
    scale: List[float]
    type: str = "scale"
    name: Optional[str] = None
    input: Optional[Union[None, str, CoordinateSystem]]
    output: Optional[Union[None, str, CoordinateSystem]]


@dataclass
class Translation:
    translation: List[float]
    type: str = "translation"
    name: Optional[str] = None
    input: Optional[Union[None, str, CoordinateSystem]]
    output: Optional[Union[None, str, CoordinateSystem]]


coordinateTransformations = Union[Scale, Translation]


@dataclass
class TransformSequence:
    input: Optional[Union[str, CoordinateSystem]]
    output: Optional[Union[str, CoordinateSystem]]
    transformations: List[coordinateTransformations]
    type: str = "sequence"
    name: Optional[str] = None


@dataclass
class Dataset:
    path: str
    coordinateTransformations: Optional[Union[coordinateTransformations, TransformSequence]]


@dataclass
class Metadata:
    datasets: List[Dataset]
    coordinateSystems: List[CoordinateSystem]
    coordinateTransformations: Optional[List[Union[Scale, Translation, TransformSequence]]] = None
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None