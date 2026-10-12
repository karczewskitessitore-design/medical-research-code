"""Verify GitHub Release split assets and extract them without joining on disk."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


class MultiPartReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Only bounded streaming reads are supported')
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            data = self.current.read(remaining)
            if not data:
                self.current.close()
                self.current = None
                continue
            chunks.append(data)
            remaining -= len(data)
        return b''.join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore(parts_dir, category, output, checksum_file):
    assets = json.loads(checksum_file.read_text(encoding='utf-8'))
    if isinstance(assets, dict):
        assets = assets['assets']
    expected = sorted((a for a in assets if a['category'] == category and '.tar.gz.part' in a['name']), key=lambda a: a['name'])
    if not expected:
        raise ValueError('No archive parts for the requested category')
    paths = []
    for number, asset in enumerate(expected, 1):
        if asset['name'] != f'{category}.tar.gz.part{number:04d}':
            raise ValueError('Archive parts are missing or out of sequence')
        path = parts_dir / asset['name']
        if path.stat().st_size != asset['bytes']:
            raise ValueError('Wrong file size: ' + path.name)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while data := handle.read(8 * 1024 * 1024):
                digest.update(data)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('Checksum mismatch: ' + path.name)
        print('Verified', path.name, flush=True)
        paths.append(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    with MultiPartReader(paths) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                if member.issym() or not (member.isfile() or member.islnk() or member.isdir()):
                    raise ValueError('Unexpected archive entry: ' + member.name)
                (output / member.name).resolve().relative_to(output)
                if member.islnk():
                    (output / member.linkname).resolve().relative_to(output)
                if hasattr(tarfile, 'data_filter'):
                    archive.extract(member, path=output, filter='data')
                else:
                    archive.extract(member, path=output)
    print('Restored', category, 'to', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', required=True, type=Path)
    parser.add_argument('--category', required=True, choices=('image_data', 'model_weights', 'environment'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checksums', required=True, type=Path)
    args = parser.parse_args()
    restore(args.parts_dir, args.category, args.output, args.checksums)
