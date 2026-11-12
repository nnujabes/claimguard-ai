"""Verify all three real models through the same pipeline as the Streamlit UI."""
import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.ui_pipeline import analyze_image, prepare_uploads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, help='Optional JPEG/PNG test image')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = args.image
        if source is None:
            source = root / 'sample.png'
            image = Image.new('RGB', (640, 420), (72, 86, 96))
            ImageDraw.Draw(image).rectangle((120, 150, 530, 330), fill=(38, 96, 170))
            image.save(source)
        uploaded = prepare_uploads([{'name': source.name,
                                    'data': base64.b64encode(source.read_bytes()).decode()}], root / 'uploads')[0]
        result = analyze_image(uploaded['path'], uploaded['name'],
                               lambda model, _: print(f'Running {model}...', flush=True))
        print(json.dumps({key: result[key] for key in ('complete', 'scores', 'score', 'route', 'models')},
                         ensure_ascii=False, indent=2))
        if not result['complete']:
            raise SystemExit('FAIL: one or more actual model inferences failed')
        print('PASS: all three official model inferences completed')


if __name__ == '__main__':
    main()
