// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * ITK-Wasm downsampling support for multiscale generation
 */

import type { Image } from "itk-wasm";
import {
  downsampleBinShrinkNode as downsampleBinShrink,
  downsampleLabelImageNode as downsampleLabelImage,
  downsampleNode as downsample,
  gaussianKernelRadiusNode as gaussianKernelRadius,
} from "@itk-wasm/downsample";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

interface DimFactors {
  [key: string]: number;
}

/**
 * Convert dimension scale factors to ITK-Wasm format
 */
function dimScaleFactors(
  dims: string[],
  scaleFactor: Record<string, number> | number,
  previousDimFactors: DimFactors
): DimFactors {
  const dimFactors: DimFactors = {};

  if (typeof scaleFactor === "number") {
    for (const dim of dims) {
      if (SPATIAL_DIMS.includes(dim)) {
        dimFactors[dim] = scaleFactor;
      } else {
        dimFactors[dim] = previousDimFactors[dim] || 1;
      }
    }
  } else {
    for (const dim of dims) {
      if (dim in scaleFactor) {
        dimFactors[dim] = scaleFactor[dim];
      } else {
        dimFactors[dim] = previousDimFactors[dim] || 1;
      }
    }
  }

  return dimFactors;
}

/**
 * Update previous dimension factors
 */
function updatePreviousDimFactors(
  scaleFactor: Record<string, number> | number,
  spatialDims: string[],
  previousDimFactors: DimFactors
): DimFactors {
  const updated: DimFactors = { ...previousDimFactors };

  if (typeof scaleFactor === "number") {
    for (const dim of spatialDims) {
      updated[dim] = scaleFactor;
    }
  } else {
    for (const dim of spatialDims) {
      if (dim in scaleFactor) {
        updated[dim] = scaleFactor[dim];
      }
    }
  }

  return updated;
}

/**
 * Compute next scale metadata
 */
function nextScaleMetadata(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[]
): [Record<string, number>, Record<string, number>] {
  const translation: Record<string, number> = {};
  const scale: Record<string, number> = {};

  for (const dim of image.dims) {
    if (spatialDims.includes(dim)) {
      const factor = dimFactors[dim];
      scale[dim] = image.scale[dim] * factor;
      translation[dim] =
        image.translation[dim] + 0.5 * (factor - 1) * image.scale[dim];
    } else {
      // Only copy non-spatial dimensions if they exist in the source
      if (dim in image.scale) {
        scale[dim] = image.scale[dim];
      }
      if (dim in image.translation) {
        translation[dim] = image.translation[dim];
      }
    }
  }

  return [translation, scale];
}

/**
 * Compute Gaussian kernel sigma values in pixel units for downsampling.
 *
 * Formula: sigma = sqrt((k^2 - 1^2)/(2*sqrt(2*ln(2)))^2)
 *
 * Reference:
 * - https://discourse.itk.org/t/resampling-to-isotropic-signal-processing-theory/1403/16
 * - https://doi.org/10.1007/978-3-319-24571-3_81
 * - http://discovery.ucl.ac.uk/1469251/1/scale-factor-point-5.pdf
 *
 * @param shrinkFactors - Shrink ratio along each axis
 * @returns Standard deviation of Gaussian kernel along each axis
 */
function computeSigma(shrinkFactors: number[]): number[] {
  const denominator = Math.pow(2 * Math.sqrt(2 * Math.log(2)), 2);
  return shrinkFactors.map((factor) =>
    Math.sqrt((factor * factor - 1) / denominator)
  );
}

/**
 * Convert zarr array to ITK-Wasm Image format
 * If isVector is true, ensures "c" dimension is last by transposing if needed
 */
async function zarrToItkImage(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  dims: string[],
  isVector = false
): Promise<Image> {
  // Read the full array data
  const result = await zarr.get(array);

  // Ensure we have the data
  if (!result.data || result.data.length === 0) {
    throw new Error("Zarr array data is empty");
  }

  let data:
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;
  let shape = result.shape;
  let _finalDims = dims;

  // If vector image, ensure "c" is last dimension
  if (isVector) {
    const cIndex = dims.indexOf("c");
    if (cIndex !== -1 && cIndex !== dims.length - 1) {
      // Need to transpose to move "c" to the end
      const permutation = dims.map((_, i) => i).filter((i) => i !== cIndex);
      permutation.push(cIndex);

      // Reorder dims
      _finalDims = permutation.map((i) => dims[i]);

      // Reorder shape
      shape = permutation.map((i) => result.shape[i]);

      // Transpose the data
      data = transposeArray(
        result.data,
        result.shape,
        permutation,
        getItkComponentType(result.data)
      );
    } else {
      // "c" already at end or not present, just copy data
      data = copyTypedArray(result.data);
    }
  } else {
    // Not a vector image, just copy data
    data = copyTypedArray(result.data);
  }

  // For vector images, the last dimension is the component count, not a spatial dimension
  const spatialShape = isVector ? shape.slice(0, -1) : shape;
  const components = isVector ? shape[shape.length - 1] : 1;

  // Create ITK-Wasm image
  const itkImage: Image = {
    imageType: {
      dimension: spatialShape.length,
      componentType: getItkComponentType(data),
      pixelType: isVector ? "VariableLengthVector" : "Scalar",
      components,
    },
    name: "image",
    origin: spatialShape.map(() => 0),
    spacing: spatialShape.map(() => 1),
    direction: createIdentityMatrix(spatialShape.length),
    size: spatialShape as number[],
    data,
    metadata: new Map(),
  };

  return itkImage;
}

/**
 * Copy typed array to appropriate type
 */
function copyTypedArray(
  data: unknown
):
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array
  | BigInt64Array
  | BigUint64Array {
  if (data instanceof Float32Array) {
    return new Float32Array(data);
  } else if (data instanceof Float64Array) {
    return new Float64Array(data);
  } else if (data instanceof Uint8Array) {
    return new Uint8Array(data);
  } else if (data instanceof Int8Array) {
    return new Int8Array(data);
  } else if (data instanceof Uint16Array) {
    return new Uint16Array(data);
  } else if (data instanceof Int16Array) {
    return new Int16Array(data);
  } else if (data instanceof Uint32Array) {
    return new Uint32Array(data);
  } else if (data instanceof Int32Array) {
    return new Int32Array(data);
  } else if (data instanceof BigInt64Array) {
    return new BigInt64Array(data);
  } else if (data instanceof BigUint64Array) {
    return new BigUint64Array(data);
  } else {
    // Convert to Float32Array as fallback
    return new Float32Array(data as ArrayLike<number>);
  }
}

/**
 * Transpose array data according to permutation
 */
function transposeArray(
  data: unknown,
  shape: number[],
  permutation: number[],
  componentType:
    | "uint8"
    | "int8"
    | "uint16"
    | "int16"
    | "uint32"
    | "int32"
    | "uint64"
    | "int64"
    | "float32"
    | "float64"
):
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array
  | BigInt64Array
  | BigUint64Array {
  const typedData = data as
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;

  // Create output array of same type
  let output:
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;
  const totalSize = typedData.length;

  switch (componentType) {
    case "uint8":
      output = new Uint8Array(totalSize);
      break;
    case "int8":
      output = new Int8Array(totalSize);
      break;
    case "int16":
      output = new Int16Array(totalSize);
      break;
    case "uint16":
      output = new Uint16Array(totalSize);
      break;
    case "int32":
      output = new Int32Array(totalSize);
      break;
    case "uint32":
      output = new Uint32Array(totalSize);
      break;
    case "int64":
      output = new BigInt64Array(totalSize);
      break;
    case "uint64":
      output = new BigUint64Array(totalSize);
      break;
    case "float64":
      output = new Float64Array(totalSize);
      break;
    case "float32":
    default:
      output = new Float32Array(totalSize);
      break;
  }

  // Calculate strides for source
  const sourceStride = calculateStride(shape);

  // Calculate new shape after permutation
  const newShape = permutation.map((i) => shape[i]);
  const targetStride = calculateStride(newShape);

  // Perform transpose
  const indices = new Array(shape.length).fill(0);

  for (let i = 0; i < totalSize; i++) {
    // Calculate source index from multi-dimensional indices
    let sourceIdx = 0;
    for (let j = 0; j < shape.length; j++) {
      sourceIdx += indices[j] * sourceStride[j];
    }

    // Calculate target index with permuted dimensions
    let targetIdx = 0;
    for (let j = 0; j < permutation.length; j++) {
      targetIdx += indices[permutation[j]] * targetStride[j];
    }

    output[targetIdx] = typedData[sourceIdx];

    // Increment indices
    for (let j = shape.length - 1; j >= 0; j--) {
      indices[j]++;
      if (indices[j] < shape[j]) break;
      indices[j] = 0;
    }
  }

  return output;
}

/**
 * Get ITK component type from typed array
 */
function getItkComponentType(
  data: unknown
):
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32"
  | "uint64"
  | "int64"
  | "float32"
  | "float64" {
  if (data instanceof Uint8Array) return "uint8";
  if (data instanceof Int8Array) return "int8";
  if (data instanceof Uint16Array) return "uint16";
  if (data instanceof Int16Array) return "int16";
  if (data instanceof Uint32Array) return "uint32";
  if (data instanceof Int32Array) return "int32";
  if (data instanceof BigUint64Array) return "uint64";
  if (data instanceof BigInt64Array) return "int64";
  if (data instanceof Float64Array) return "float64";
  return "float32";
}

/**
 * Create identity matrix for ITK direction
 */
function createIdentityMatrix(dimension: number): Float64Array {
  const matrix = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension * dimension; i++) {
    matrix[i] = i % (dimension + 1) === 0 ? 1 : 0;
  }
  return matrix;
}

/**
 * Convert ITK-Wasm Image back to zarr array
 */
async function itkImageToZarr(
  itkImage: Image,
  path: string,
  chunkShape: number[]
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  // Use in-memory store
  const store: Map<string, Uint8Array> = new Map();
  const root = zarr.root(store);

  // Determine data type
  let dataType: zarr.DataType;
  if (itkImage.data instanceof Uint8Array) {
    dataType = "uint8";
  } else if (itkImage.data instanceof Int8Array) {
    dataType = "int8";
  } else if (itkImage.data instanceof Int16Array) {
    dataType = "int16";
  } else if (itkImage.data instanceof Uint16Array) {
    dataType = "uint16";
  } else if (itkImage.data instanceof Int32Array) {
    dataType = "int32";
  } else if (itkImage.data instanceof Uint32Array) {
    dataType = "uint32";
  } else if (itkImage.data instanceof BigInt64Array) {
    dataType = "int64";
  } else if (itkImage.data instanceof BigUint64Array) {
    dataType = "uint64";
  } else if (itkImage.data instanceof Float64Array) {
    dataType = "float64";
  } else if (itkImage.data instanceof Float32Array) {
    dataType = "float32";
  } else {
    dataType = "float32";
  }

  const array = await zarr.create(root.resolve(path), {
    shape: itkImage.size,
    chunk_shape: chunkShape,
    data_type: dataType,
    fill_value: 0,
  });

  // Write data
  await zarr.set(array, [], {
    data: itkImage.data as Float32Array,
    shape: itkImage.size,
    stride: calculateStride(itkImage.size),
  });

  return array;
}

/**
 * Calculate stride for array
 */
function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

/**
 * Process channel-first data by downsampling each channel separately
 */
async function downsampleChannelFirst(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  smoothing: "gaussian" | "bin_shrink" | "label_image"
): Promise<NgffImage> {
  // Get the channel index and count
  const cIndex = image.dims.indexOf("c");
  const result = await zarr.get(image.data);
  const channelCount = result.shape[cIndex];

  // Process each channel separately
  const downsampledChannels: zarr.Array<zarr.DataType, zarr.Readable>[] = [];

  for (let channelIdx = 0; channelIdx < channelCount; channelIdx++) {
    // Extract single channel data
    const channelSlice = extractChannel(result, cIndex, channelIdx);

    // Create temporary zarr array for this channel
    const store: Map<string, Uint8Array> = new Map();
    const root = zarr.root(store);

    const channelDims = image.dims.filter((d) => d !== "c");
    const channelShape = result.shape.filter((_, i) => i !== cIndex);
    const chunkShape = channelShape.map((s) => Math.min(s, 256));

    const channelArray = await zarr.create(root.resolve("channel"), {
      shape: channelShape,
      chunk_shape: chunkShape,
      data_type: getItkComponentType(result.data),
      fill_value: 0,
    });

    await zarr.set(channelArray, [], {
      data: channelSlice,
      shape: channelShape,
      stride: calculateStride(channelShape),
    });

    // Create NgffImage for this channel (unused but kept for potential future use)
    // const _channelImage = new NgffImage({
    //   data: channelArray,
    //   dims: channelDims,
    //   scale: Object.fromEntries(
    //     Object.entries(image.scale).filter(([k]) => k !== "c")
    //   ),
    //   translation: Object.fromEntries(
    //     Object.entries(image.translation).filter(([k]) => k !== "c")
    //   ),
    //   name: image.name,
    //   axesUnits: image.axesUnits,
    //   computedCallbacks: image.computedCallbacks,
    // });

    // Downsample this channel
    const itkImage = await zarrToItkImage(channelArray, channelDims, false);

    const shrinkFactors: number[] = [];
    for (let i = 0; i < channelDims.length; i++) {
      const dim = channelDims[i];
      if (SPATIAL_DIMS.includes(dim)) {
        shrinkFactors.push(dimFactors[dim] || 1);
      } else {
        shrinkFactors.push(1); // Non-spatial dimensions don't shrink
      }
    }

    let downsampled: Image;

    if (smoothing === "gaussian") {
      const blockSize = itkImage.size.slice().reverse();
      const sigma = computeSigma(shrinkFactors);
      const { radius: _radius } = await gaussianKernelRadius({
        size: blockSize,
        sigma,
      });

      const result = await downsample(itkImage, {
        shrinkFactors,
        cropRadius: shrinkFactors.map(() => 0),
      });
      downsampled = result.downsampled;
    } else if (smoothing === "bin_shrink") {
      const result = await downsampleBinShrink(itkImage, {
        shrinkFactors,
      });
      downsampled = result.downsampled;
    } else if (smoothing === "label_image") {
      const blockSize = itkImage.size.slice().reverse();
      const sigma = computeSigma(shrinkFactors);
      const { radius: _radius } = await gaussianKernelRadius({
        size: blockSize,
        sigma,
      });

      const result = await downsampleLabelImage(itkImage, {
        shrinkFactors,
        cropRadius: shrinkFactors.map(() => 0),
      });
      downsampled = result.downsampled;
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    // Convert back to zarr array
    const downsampledChunkShape = downsampled.size.map((s) => Math.min(s, 256));
    const downsampledArray = await itkImageToZarr(
      downsampled,
      "downsampled_channel",
      downsampledChunkShape
    );
    downsampledChannels.push(downsampledArray);
  }

  // Combine all channels back together
  const combinedArray = await combineChannels(
    downsampledChannels,
    cIndex,
    image.dims
  );

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims
  );

  return new NgffImage({
    data: combinedArray,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Extract a single channel from the data
 */
function extractChannel(
  result: { data: unknown; shape: number[] },
  cIndex: number,
  channelIdx: number
):
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array
  | BigInt64Array
  | BigUint64Array {
  const typedData = result.data as
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;
  const shape = result.shape;

  // Calculate output size (all dims except channel)
  const outputSize = shape.reduce(
    (acc, s, i) => (i === cIndex ? acc : acc * s),
    1
  );

  let output:
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;
  if (typedData instanceof Uint8Array) {
    output = new Uint8Array(outputSize);
  } else if (typedData instanceof Int8Array) {
    output = new Int8Array(outputSize);
  } else if (typedData instanceof Int16Array) {
    output = new Int16Array(outputSize);
  } else if (typedData instanceof Uint16Array) {
    output = new Uint16Array(outputSize);
  } else if (typedData instanceof Int32Array) {
    output = new Int32Array(outputSize);
  } else if (typedData instanceof Uint32Array) {
    output = new Uint32Array(outputSize);
  } else if (typedData instanceof BigInt64Array) {
    output = new BigInt64Array(outputSize);
  } else if (typedData instanceof BigUint64Array) {
    output = new BigUint64Array(outputSize);
  } else if (typedData instanceof Float64Array) {
    output = new Float64Array(outputSize);
  } else {
    output = new Float32Array(outputSize);
  }

  // Calculate strides
  const stride = calculateStride(shape);
  const outputShape = shape.filter((_, i) => i !== cIndex);
  const _outputStride = calculateStride(outputShape);

  // Extract channel
  const indices = new Array(shape.length).fill(0);
  let outputIdx = 0;

  for (let i = 0; i < outputSize; i++) {
    // Set channel index
    indices[cIndex] = channelIdx;

    // Calculate source index
    let sourceIdx = 0;
    for (let j = 0; j < shape.length; j++) {
      sourceIdx += indices[j] * stride[j];
    }

    output[outputIdx++] = typedData[sourceIdx];

    // Increment indices (skip channel dimension)
    for (let j = shape.length - 1; j >= 0; j--) {
      if (j === cIndex) continue;
      indices[j]++;
      if (indices[j] < shape[j]) break;
      indices[j] = 0;
    }
  }

  return output;
}

/**
 * Combine multiple channel arrays back into a single multi-channel array
 */
async function combineChannels(
  channels: zarr.Array<zarr.DataType, zarr.Readable>[],
  cIndex: number,
  _originalDims: string[]
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  // Read all channel data
  const channelData = await Promise.all(channels.map((c) => zarr.get(c)));

  // Determine combined shape
  const firstChannel = channelData[0];
  const channelShape = firstChannel.shape;
  const combinedShape = [...channelShape];
  combinedShape.splice(cIndex, 0, channels.length);

  // Create combined array
  const store: Map<string, Uint8Array> = new Map();
  const root = zarr.root(store);

  const chunkShape = combinedShape.map((s) => Math.min(s, 256));
  const dataType = getItkComponentType(firstChannel.data);

  const combinedArray = await zarr.create(root.resolve("combined"), {
    shape: combinedShape,
    chunk_shape: chunkShape,
    data_type: dataType,
    fill_value: 0,
  });

  // Combine all channels
  const totalSize = combinedShape.reduce((acc, s) => acc * s, 1);
  let combined:
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array;

  if (dataType === "uint8") {
    combined = new Uint8Array(totalSize);
  } else if (dataType === "int8") {
    combined = new Int8Array(totalSize);
  } else if (dataType === "int16") {
    combined = new Int16Array(totalSize);
  } else if (dataType === "uint16") {
    combined = new Uint16Array(totalSize);
  } else if (dataType === "int32") {
    combined = new Int32Array(totalSize);
  } else if (dataType === "uint32") {
    combined = new Uint32Array(totalSize);
  } else if (dataType === "int64") {
    combined = new BigInt64Array(totalSize);
  } else if (dataType === "uint64") {
    combined = new BigUint64Array(totalSize);
  } else if (dataType === "float64") {
    combined = new Float64Array(totalSize);
  } else {
    combined = new Float32Array(totalSize);
  }

  const stride = calculateStride(combinedShape);
  const _channelStride = calculateStride(channelShape);

  // Copy each channel's data
  for (let c = 0; c < channels.length; c++) {
    const channelTypedData = channelData[c].data as typeof combined;
    const indices = new Array(combinedShape.length).fill(0);

    for (let i = 0; i < channelTypedData.length; i++) {
      // Set channel index
      indices[cIndex] = c;

      // Calculate target index in combined array
      let targetIdx = 0;
      for (let j = 0; j < combinedShape.length; j++) {
        targetIdx += indices[j] * stride[j];
      }

      combined[targetIdx] = channelTypedData[i];

      // Increment indices (skip channel dimension)
      for (let j = combinedShape.length - 1; j >= 0; j--) {
        if (j === cIndex) continue;
        indices[j]++;
        if (indices[j] < combinedShape[j]) break;
        indices[j] = 0;
      }
    }
  }

  // Write combined data
  await zarr.set(combinedArray, [], {
    data: combined as Float32Array,
    shape: combinedShape,
    stride,
  });

  return combinedArray;
}

/**
 * Perform Gaussian downsampling using ITK-Wasm
 */
async function downsampleGaussian(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[]
): Promise<NgffImage> {
  const cIndex = image.dims.indexOf("c");
  const isVector = cIndex === image.dims.length - 1;
  const isChannelFirst =
    cIndex !== -1 && cIndex < image.dims.length - 1 && !isVector;

  // If channel is first (before spatial dims), process each channel separately
  if (isChannelFirst) {
    return await downsampleChannelFirst(
      image,
      dimFactors,
      spatialDims,
      "gaussian"
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for spatial dimensions only
  // For vector images, the last dimension (c) is NOT a spatial dimension in the ITK image
  const shrinkFactors: number[] = [];
  const effectiveDims = isVector ? image.dims.slice(0, -1) : image.dims;

  for (let i = 0; i < effectiveDims.length; i++) {
    const dim = effectiveDims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Compute kernel radius - sigma should also be for ALL dimensions
  const blockSize = itkImage.size.slice().reverse();
  const sigma = computeSigma(shrinkFactors);
  const { radius: _radius } = await gaussianKernelRadius({
    size: blockSize,
    sigma,
  });

  // Perform downsampling
  const { downsampled } = await downsample(itkImage, {
    shrinkFactors,
    cropRadius: shrinkFactors.map(() => 0),
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims
  );

  // Convert back to zarr array
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(downsampled, "downsampled", chunkShape);

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform bin shrink downsampling using ITK-Wasm
 */
async function downsampleBinShrinkImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[]
): Promise<NgffImage> {
  const cIndex = image.dims.indexOf("c");
  const isVector = cIndex === image.dims.length - 1;
  const isChannelFirst =
    cIndex !== -1 && cIndex < image.dims.length - 1 && !isVector;

  // If channel is first (before spatial dims), process each channel separately
  if (isChannelFirst) {
    return await downsampleChannelFirst(
      image,
      dimFactors,
      spatialDims,
      "bin_shrink"
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for spatial dimensions only
  // For vector images, the last dimension (c) is NOT a spatial dimension in the ITK image
  const shrinkFactors: number[] = [];
  const effectiveDims = isVector ? image.dims.slice(0, -1) : image.dims;

  for (let i = 0; i < effectiveDims.length; i++) {
    const dim = effectiveDims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Perform downsampling
  const { downsampled } = await downsampleBinShrink(itkImage, {
    shrinkFactors,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims
  );

  // Convert back to zarr array
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(downsampled, "downsampled", chunkShape);

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform label image downsampling using ITK-Wasm
 */
async function downsampleLabelImageImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[]
): Promise<NgffImage> {
  const cIndex = image.dims.indexOf("c");
  const isVector = cIndex === image.dims.length - 1;
  const isChannelFirst =
    cIndex !== -1 && cIndex < image.dims.length - 1 && !isVector;

  // If channel is first (before spatial dims), process each channel separately
  if (isChannelFirst) {
    return await downsampleChannelFirst(
      image,
      dimFactors,
      spatialDims,
      "label_image"
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for spatial dimensions only
  // For vector images, the last dimension (c) is NOT a spatial dimension in the ITK image
  const shrinkFactors: number[] = [];
  const effectiveDims = isVector ? image.dims.slice(0, -1) : image.dims;

  for (let i = 0; i < effectiveDims.length; i++) {
    const dim = effectiveDims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Compute kernel radius
  const blockSize = itkImage.size.slice().reverse();
  const sigma = computeSigma(shrinkFactors);
  const { radius: _radius } = await gaussianKernelRadius({
    size: blockSize,
    sigma,
  });

  // Perform downsampling
  const { downsampled } = await downsampleLabelImage(itkImage, {
    shrinkFactors,
    cropRadius: shrinkFactors.map(() => 0),
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims
  );

  // Convert back to zarr array
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(downsampled, "downsampled", chunkShape);

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Main downsampling function for ITK-Wasm
 */
export async function downsampleItkWasm(
  ngffImage: NgffImage,
  scaleFactors: (Record<string, number> | number)[],
  smoothing: "gaussian" | "bin_shrink" | "label_image"
): Promise<NgffImage[]> {
  const multiscales: NgffImage[] = [ngffImage];
  let previousImage = ngffImage;
  const dims = ngffImage.dims;
  let previousDimFactors: DimFactors = {};
  for (const dim of dims) {
    previousDimFactors[dim] = 1;
  }

  const spatialDims = dims.filter((dim) => SPATIAL_DIMS.includes(dim));

  for (const scaleFactor of scaleFactors) {
    const dimFactors = dimScaleFactors(dims, scaleFactor, previousDimFactors);
    previousDimFactors = updatePreviousDimFactors(
      scaleFactor,
      spatialDims,
      previousDimFactors
    );

    let downsampled: NgffImage;
    if (smoothing === "gaussian") {
      downsampled = await downsampleGaussian(
        previousImage,
        dimFactors,
        spatialDims
      );
    } else if (smoothing === "bin_shrink") {
      downsampled = await downsampleBinShrinkImpl(
        previousImage,
        dimFactors,
        spatialDims
      );
    } else if (smoothing === "label_image") {
      downsampled = await downsampleLabelImageImpl(
        previousImage,
        dimFactors,
        spatialDims
      );
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    multiscales.push(downsampled);
    previousImage = downsampled;
  }

  return multiscales;
}
