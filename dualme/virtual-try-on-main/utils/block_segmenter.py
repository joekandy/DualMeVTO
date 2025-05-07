import torch

def segment_weights(weight, segments=4, dim=0):
    if weight.shape[dim] % segments != 0:
        raise ValueError("Weight dimension not divisible by number of segments")
    chunk_size = weight.shape[dim] // segments
    return torch.chunk(weight, segments, dim=dim)

def average_segments(seg_a, seg_b):
    if len(seg_a) != len(seg_b):
        raise ValueError("Segments must have same length")
    return [0.5 * (a + b) for a, b in zip(seg_a, seg_b)]

def merge_segments(segments, dim=0):
    return torch.cat(segments, dim=dim)

def interpolate_blocks(block_a, block_b, segments=4, dim=0):
    seg_a = segment_weights(block_a, segments, dim=dim)
    seg_b = segment_weights(block_b, segments, dim=dim)
    interpolated = average_segments(seg_a, seg_b)
    return merge_segments(interpolated, dim=dim)
