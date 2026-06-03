def what(file, h=None):
    if h is None:
        with open(file, 'rb') as f:
            h = f.read(32)

    if h.startswith(b'\xff\xd8'):
        return 'jpeg'
    if h.startswith(b'\x89PNG'):
        return 'png'
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if h.startswith(b'BM'):
        return 'bmp'
    if h.startswith(b'II*\x00') or h.startswith(b'MM\x00*'):
        return 'tiff'

    return None