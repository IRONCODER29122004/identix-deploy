import os

import torch


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
    return ckpt


def main() -> int:
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth'))
    print('ckpt_path:', ckpt_path)
    print('exists:', os.path.exists(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = _unwrap_state_dict(ckpt)

    print('ckpt type:', type(ckpt))
    print('state_dict type:', type(state_dict))
    if not isinstance(state_dict, dict):
        print('Unexpected checkpoint format (not a dict).')
        return 2

    keys = list(state_dict.keys())
    print('num_keys:', len(keys))
    print('first_10_keys:', keys[:10])

    want = [
        'cp.arm16.conv.conv.weight',
        'cp.arm32.conv.conv.weight',
        'cp.conv_head16.conv.weight',
        'cp.conv_head32.conv.weight',
        'cp.conv_avg.conv.weight',
        'sp.conv_out.conv.weight',
        'ffm.convblk.conv.weight',
        'conv_out.conv.conv.weight',
        'conv_out.conv_out.weight',
        'conv_out16.conv.conv.weight',
        'conv_out16.conv_out.weight',
        'conv_out32.conv.conv.weight',
        'conv_out32.conv_out.weight',
    ]
    for k in want:
        v = state_dict.get(k)
        print(k, '->', None if v is None else tuple(v.shape))

    for prefix in ['cp.', 'sp.', 'ffm.', 'conv_out', 'conv_out16', 'conv_out32']:
        sample = [k for k in keys if k.startswith(prefix)][:12]
        print(f'sample {prefix}:', sample)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
