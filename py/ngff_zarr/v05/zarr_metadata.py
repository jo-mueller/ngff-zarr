# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List, Optional
from dataclasses import dataclass

from ..v04.zarr_metadata import Axis, Transform, Dataset, Omero, MethodMetadata
from .._supported_versions import SUPPORTED_VERSIONS


@dataclass
class Metadata:
    axes: List[Axis]
    datasets: List[Dataset]
    coordinateTransformations: Optional[List[Transform]]
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None

    def to_version(self, version: str) -> "Metadata":
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version conversion: 0.5 -> {version}")
        if version == "0.5":
            return self
        elif version == "0.4":
            return self.to_v04()
            
        
    @classmethod
    def from_version(cls, metadata: "Metadata") -> "Metadata":
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        
        if isinstance(metadata, Metadata_v04):
            return cls.from_v04(metadata)
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def to_v04(self) -> "Metadata":
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        
        metadata = Metadata_v04(
            axes=self.axes,
            datasets=self.datasets,
            coordinateTransformations=self.coordinateTransformations,
            name=self.name,
            metadata=self.metadata,
            type=self.type,
            omero=self.omero,
        )
        return metadata
    
    @classmethod
    def from_v04(cls, metadata_v04: "Metadata") -> "Metadata":
        
        metadata = cls(
            axes=metadata_v04.axes,
            datasets=metadata_v04.datasets,
            coordinateTransformations=metadata_v04.coordinateTransformations,
            name=metadata_v04.name,
            metadata=metadata_v04.metadata,
            type=metadata_v04.type,
            omero=metadata_v04.omero,
        )
        return metadata
    
    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])