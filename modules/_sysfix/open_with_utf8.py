import builtins


def open_with_utf8(file, mode='r', *args, **kwargs):
    if 'b' in mode:
        return _open(file, mode, *args, **kwargs)
    else:
        return _open(file, mode, encoding='utf-8', *args, **kwargs)


_open = builtins.open
builtins.open = open_with_utf8
