from .block_extractor import extract_conv_blocks
from .block_matcher import match_blocks
from .block_segmenter import interpolate_blocks

def hyperclone(model, segments=4, threshold=0.9):
    cloned_model = model.__class__()  # crea nuova istanza
    cloned_model.load_state_dict(model.state_dict(), strict=False)

    conv_blocks = extract_conv_blocks(model)
    matches = match_blocks(conv_blocks, threshold=threshold)

    for (name_a, block_a), (name_b, block_b) in matches:
        interpolated_weight = interpolate_blocks(
            block_a.weight.data, block_b.weight.data, segments=segments, dim=0
        )
        interpolated_bias = None
        if block_a.bias is not None and block_b.bias is not None:
            interpolated_bias = 0.5 * (block_a.bias.data + block_b.bias.data)

        setattr(dict(cloned_model.named_modules())[name_a], 'weight', interpolated_weight)
        if interpolated_bias is not None:
            setattr(dict(cloned_model.named_modules())[name_a], 'bias', interpolated_bias)

    return cloned_model
