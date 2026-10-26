"""Convert Basic Pitch's released Keras weights for ``processing.music.frontend``.

    uv run --with tensorflow==2.16.2 python -m processing.tools.convert_basic_pitch

The basic-pitch package ships its ICASSP 2022 checkpoint as a TensorFlow
SavedModel, a Core ML package, TFLite and ONNX. The SavedModel's variables are
the only copy with the batch-norm layers still separate from the convolutions
they follow, which fine-tuning needs. Reading them needs TensorFlow, which
inlay does not otherwise depend on, so this is run once in a throwaway
environment and writes an ordinary PyTorch checkpoint.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from processing.music import frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=frontend.SHIPPED)
    args = parser.parse_args()

    import tensorflow as tf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    path = str(Path(ICASSP_2022_MODEL_PATH)/'variables'/'variables')
    reader = tf.train.load_checkpoint(path)
    variables = {name: np.asarray(reader.get_tensor(name))
                 for name, _ in tf.train.list_variables(path)
                 if name.startswith('layer_with_weights-')}
    state = frontend.keras_state_dict(variables)
    model = frontend.BasicPitch()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the CQT's fixed buffers and batch-norm bookkeeping are not in Keras.
    missing = [k for k in missing if not k.startswith('cqt.') and not k.endswith('num_batches_tracked')]
    if missing or unexpected:
        raise RuntimeError(f'weight mapping failed: missing {missing}, unexpected {unexpected}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'source': 'basic-pitch 0.4.0 icassp_2022'}, args.output)
    print(f'wrote {args.output}: {sum(p.numel() for p in model.parameters())} weights')


if __name__ == '__main__':
    main()
