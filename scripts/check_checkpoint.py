import os
import sys
import torch


def main() -> int:
    # Ensure the Required/ folder is importable when running from Required/scripts
    required_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if required_dir not in sys.path:
        sys.path.insert(0, required_dir)

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
    ckpt_path = os.path.abspath(ckpt_path)

    print('ckpt exists:', os.path.exists(ckpt_path), ckpt_path)

    # Import the ResNet50-based BiSeNet used by the Flask app
    from landmark_app import BiSeNet  # type: ignore

    device = torch.device('cpu')
    model = BiSeNet(11).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('ckpt type:', type(ckpt))

    state_dict = ckpt
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']

    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('missing keys:', len(missing))
    print('unexpected keys:', len(unexpected))
    print('sample missing:', missing[:20])
    print('sample unexpected:', unexpected[:20])

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    if isinstance(y, tuple):
        print('forward tuple lens:', len(y))
        print('forward shapes:', [t.shape for t in y])
    else:
        print('forward shape:', y.shape)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
