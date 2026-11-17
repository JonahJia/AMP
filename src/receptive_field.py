from typing import List, Dict, Tuple
import math

def compute_layer_rf_info(
    layer_filter_size: int,
    layer_stride: int,
    layer_padding: int,
    previous_rf_info: List[float]
) -> List[float]:
    """
    Compute receptive field info for a layer.
    
    Args:
        layer_filter_size: Filter/kernel size
        layer_stride: Stride
        layer_padding: Padding (or 'SAME'/'VALID')
        previous_rf_info: [n_in, j_in, r_in, start_in] from previous layer
    
    Returns:
        [n_out, j_out, r_out, start_out] for current layer
    """
    n_in, j_in, r_in, start_in = previous_rf_info
    
    # Handle padding
    if isinstance(layer_padding, str):
        if layer_padding == 'SAME':
            n_out = math.ceil(n_in / layer_stride)
            if n_in % layer_stride == 0:
                pad = max(layer_filter_size - layer_stride, 0)
            else:
                pad = max(layer_filter_size - (n_in % layer_stride), 0)
        elif layer_padding == 'VALID':
            n_out = math.ceil((n_in - layer_filter_size + 1) / layer_stride)
            pad = 0
        else:
            raise ValueError(f"Unknown padding type: {layer_padding}")
    else:
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
    
    # Compute new receptive field parameters
    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1) * j_in
    start_out = start_in + ((layer_filter_size - 1) / 2 - pad / 2) * j_in
    
    return [n_out, j_out, r_out, start_out]


def compute_proto_layer_rf_info(
    img_size: int,
    layer_filter_sizes: List[int],
    layer_strides: List[int],
    layer_paddings: List[int],
    prototype_kernel_size: int
) -> Dict[str, float]:
    """
    Compute receptive field info for prototype layer.
    
    Args:
        img_size: Input image size
        layer_filter_sizes: List of filter sizes for all layers
        layer_strides: List of strides for all layers
        layer_paddings: List of paddings for all layers
        prototype_kernel_size: Size of prototype kernel
    
    Returns:
        Dictionary with receptive field information
    """
    assert len(layer_filter_sizes) == len(layer_strides) == len(layer_paddings)
    
    # Initialize with input layer info
    rf_info = [img_size, 1, 1, 0.5]
    
    # Propagate through all layers
    for filter_size, stride, padding in zip(
        layer_filter_sizes, layer_strides, layer_paddings
    ):
        rf_info = compute_layer_rf_info(
            filter_size, stride, padding, rf_info
        )
    
    # Add prototype layer
    proto_rf_info = compute_layer_rf_info(
        prototype_kernel_size, 1, 'VALID', rf_info
    )
    
    return {
        'size': proto_rf_info[0],
        'jump': proto_rf_info[1],
        'receptive_field': proto_rf_info[2],
        'start': proto_rf_info[3]
    }


def compute_rf_prototype(
    img_size: int,
    prototype_patch_index: Tuple[int, int, int],
    proto_rf_info: Dict[str, float]
) -> Tuple[int, int, int, int, int]:
    """
    Compute receptive field of a prototype in the original image.
    
    Args:
        img_size: Original image size
        prototype_patch_index: (img_idx, height_idx, width_idx)
        proto_rf_info: Receptive field information
    
    Returns:
        (img_idx, rf_h_start, rf_h_end, rf_w_start, rf_w_end)
    """
    img_idx, height_idx, width_idx = prototype_patch_index
    
    j = proto_rf_info['jump']
    r = proto_rf_info['receptive_field']
    start = proto_rf_info['start']
    
    # Center of receptive field
    center_h = start + height_idx * j
    center_w = start + width_idx * j
    
    # Bounds
    rf_h_start = max(int(center_h - r / 2), 0)
    rf_h_end = min(int(center_h + r / 2), img_size)
    rf_w_start = max(int(center_w - r / 2), 0)
    rf_w_end = min(int(center_w + r / 2), img_size)
    
    return img_idx, rf_h_start, rf_h_end, rf_w_start, rf_w_end
