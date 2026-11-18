"""Durable per-unit synthesis identity, independent of subtitle/timing revisions."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from core.config_utils import load_key


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


@lru_cache(maxsize=2048)
def _file_hash(path, size, mtime, ctime):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def file_hash(path):
    p = Path(path).resolve(); s = p.stat()
    return _file_hash(str(p), s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def valid_audio(path):
    try:
        data, sr = sf.read(path)
        return sr > 0 and len(data) > sr * .02 and np.isfinite(data).all() and float(np.max(np.abs(data))) > 1e-6
    except (OSError, ValueError, RuntimeError):
        return False


def model_identity(cfg):
    root = Path(cfg['root']).expanduser()
    model = Path(cfg.get('model') or os.environ.get('MLX_INDEXTTS_V25_MODEL') or root / 'models/mlx-IndexTTS-2.5-8bit').expanduser()
    if not model.is_absolute(): model = root / model
    required = ['config.json', 'config.yaml', 'gpt.safetensors', 's2mel.safetensors', 'bigvgan.safetensors']
    if any(not (model / name).is_file() for name in required):
        raise FileNotFoundError(f'Fast IndexTTS2.5 model incomplete: {model}')
    files = sorted(set(model.glob('*.safetensors')) | set(model.glob('*.json')) | set(model.glob('*.yaml')) | set(model.glob('*.model')))
    return {'path': str(model.resolve()), 'files': {p.name: file_hash(p) for p in files}}


def request_identity(request, model):
    from core.tts_pronunciation import synthesis_text
    cfg = load_key('mlx_tts.backends.indextts2', {})
    return dict(version='natural-v2', text=request.text, synthesis_text=synthesis_text(request),
                language=request.language, speaker=request.speaker_id,
                reference=file_hash(request.ref_audio),
                emotion=file_hash(request.emotion_ref) if request.emotion_ref else None,
                emo_alpha=request.emo_alpha, config=cfg, model=model,
                frontend=file_hash(Path(__file__).parent/'tts_pronunciation.py'),
                reference_text=request.ref_text, native_fit=False)


MAX_ATTEMPTS = 5  # synthesis attempts per unit and identity before the ledger refuses another


class AudioLedger:
    def __init__(self, path='output/log/fast_audio_cache.json'):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {'version': 2, 'entries': {}, 'attempts': {}, 'rewrites': {}}

    def save(self): atomic_json(self.path, self.data)

    def reusable(self, key, identity):
        row = self.data['entries'].get(key, {})
        return (row.get('identity') == digest(identity) and valid_audio(row.get('path', ''))
                and file_hash(row['path']) == row.get('sha256'))

    def put(self, key, request, identity, **extra):
        if not valid_audio(request.output_path): raise ValueError('Cannot cache invalid audio')
        self.data['entries'][key] = dict(identity=digest(identity), payload=identity,
            path=str(Path(request.output_path).resolve()), sha256=file_hash(request.output_path), **extra)
        self.save()

    def reserve(self, key, identity):
        token = digest([key, identity]); count = self.data['attempts'].get(token, 0)
        if count >= MAX_ATTEMPTS: raise RuntimeError(f'Fast synthesis attempts exhausted: {key}')
        self.data['attempts'][token] = count + 1
        self.save()
