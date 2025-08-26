from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

from dask.array.core import Array as DaskArray

from .v04.zarr_metadata import Units
from .rfc4 import AnatomicalOrientation
from .rfc5 import NgffBaseTransformation, NgffCoordinateSystem, NgffSequence, NgffScale, NgffTranslation

ComputedCallback = Callable[[], None]


@dataclass
class NgffImage:
    data: DaskArray
    transformations: NgffBaseTransformation
    name: str = "image"
    axes_orientations: Optional[Mapping[str, AnatomicalOrientation]] = None
    computed_callbacks: List[ComputedCallback] = field(default_factory=list)

    @property
    def input_coordinate_system(self) -> NgffCoordinateSystem:
        return self.transformations.input_coordinate_system

    @property
    def output_coordinate_system(self) -> NgffCoordinateSystem:
        return self.transformations.output_coordinate_system
    
    @property
    def coordinate_systems(self) -> List:
        cs = [self.input_coordinate_system, self.output_coordinate_system]
        return [cs for cs in cs if cs is not None]

    @property
    def dims(self) -> Mapping[str, str]:
        if self.output_coordinate_system is not None:
            return self.output_coordinate_system.axes_names
        return None

    @property
    def axes_units(self) -> Mapping[str, str]:
        if self.output_coordinate_system is not None:
            cs = self.output_coordinate_system
            return {axis: cs.get_axis(axis).unit for axis in cs.axes_names}
        return None
    
    @property
    def scale(self) -> Mapping[str, float]:
        """
        Get legacy attribute `scale` from ngff-transformations.
        """

        if isinstance(self.transformations, NgffSequence):
            for t in self.transformations.transformations:
                if isinstance(t, NgffScale):
                    return {d: float(s) for d, s in zip(self.dims, t.scale)}
        elif isinstance(self.transformations, NgffScale):
            return {d: float(s) for d, s in zip(self.dims, self.transformations.scale)}
        return None
    
    @property
    def translation(self) -> Mapping[str, float]:
        """
        Get legacy attribute `translation` from ngff-transformations.
        """

        if isinstance(self.transformations, NgffSequence):
            for t in self.transformations.transformations:
                if isinstance(t, NgffTranslation):
                    return {d: float(s) for d, s in zip(self.dims, t.translation)}
        elif isinstance(self.transformations, NgffTranslation):
            return {d: float(s) for d, s in zip(self.dims, self.transformations.translation)}
        return {d: 0.0 for d in self.dims}