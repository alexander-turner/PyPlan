import os
import yaml
import glob
import shutil
import subprocess
import importlib
import tempfile

from toposort import toposort_flatten
from collections import deque
from contextlib import suppress
from copy import copy
from distutils._msvccompiler import MSVCCompiler, _find_exe
from .extension import WinExt


def find_directories(parent='.'):
    """Recursively find directories that contain a module.yml

    No directories should be returned that have a parent
    containing module.yml
    """
    # FIXME: This implementation does not comply with the second point

    for fname in glob.iglob(
            os.path.join(parent, '**/module.yml'), recursive=True):
        yield os.path.relpath(os.path.dirname(fname), parent)


def collect_extensions(parent='.'):
    """Collect all directories that contain a 'module.yml' recursively.

    The 'include' directory of each module is added by default to the include
    path, and 'include' directories from other extensions are added to the 
    include path if they are listed in the 'modules'
    """

    # This is somewhat confusing, because the 'modules' key in the yaml
    # files specifies the dependencies for that module, the the 'modules'
    # variable contains the actual module configurations.
    modules = {}
    dependencies = {}
    build_order = ('.i', '.mc', '.rc', '.cpp')

    for dir in find_directories(parent):
        yml = os.path.join(parent, dir, 'module.yml')
        with open(yml) as fh:
            cfg = yaml.load(fh)

        if yml is None:
            raise ValueError('Invalid configuration file')

        name = dir.replace(os.sep, '.')
        modules[name] = cfg  # Module configuration
        dependencies[name] = set(cfg.setdefault('modules',
                                                []))  # Module dependencies

    external_modules = []
    for name in toposort_flatten(dependencies):
        src = os.path.join(parent, name.replace('.', os.sep), 'src')
        sources = []

        try:
            files = os.listdir(src)
        except FileNotFoundError:
            external_modules += [
                name
            ]  # Add this to external_modules for later generation
            continue

        for item in build_order:  # Add all sources present in the 'src' directory
            for fname in files:
                if fname.endswith(item):
                    sources.append(os.path.join(src, fname))

        include_dirs = []
        f = set()
        q = deque([name])
        i = 0

        # Simple dependency resolver algorithm:
        while q:
            i += 1
            assert i < 500  # To avoid infinite loop

            dep = q.popleft().replace('.', os.sep)  # Take one module
            include_dirs.extend([  # Add the include directories
                os.path.join(parent, dep, 'src'),
                os.path.join(parent, dep, 'include')
            ])
            f.add(dep.replace(os.sep,
                              '.'))  # Add the module's dependencies to the set
            q.extend([d for d in dependencies[name] if d not in f
                      ])  # Queue modules not already in the set for processing

        cfg = modules[name]
        del cfg['modules']  # Remove the 'modules' (depenencies) key

        with suppress(KeyError):
            for i, depend in enumerate(cfg['depends']):
                cfg['depends'][i] = os.path.join(name, 'include', depend)

        for config_option in ['export_symbol_file', 'pch_header']:
            with suppress(KeyError):
                cfg[config_option] = os.path.join(
                    name.replace('.', os.sep), 'include', cfg[config_option])

        build_temp_library_dirs = [
            os.path.join(os.path.dirname(d), 'src') for d in include_dirs
        ]

        yield WinExt(
            name,
            sources=sources,
            include_dirs=copy(include_dirs),
            external_modules=copy(external_modules),
            build_temp_library_dirs=list(set(build_temp_library_dirs)),
            **cfg)

        print('collected: {}'.format(name))


def find_tools():
    """Find and return the location of the 'lib' and 'dumpbin' tools.
    
    Also returns the environment in which they should be run.
    """
    compiler = MSVCCompiler()
    compiler.initialize()

    paths = compiler._paths.split(os.pathsep)
    lib = compiler.lib
    dumpbin = _find_exe('dumpbin.exe', paths)

    env = copy(os.environ)
    env['path'] = os.pathsep.join(paths)

    return lib, dumpbin, env


def generate_libs(libs, plat_name):
    """Generate lib files to link against from python extensions."""
    tmpdir = tempfile.mkdtemp()
    lib, dumpbin, env = find_tools()

    for module in iter(importlib.import_module(lib) for lib in libs):
        pyd = module.__file__
        basename, ext = os.path.splitext(os.path.basename(pyd))

        assert ext == '.pyd'

        dll = os.path.join(tmpdir, '{}.dll'.format(basename))
        dname = os.path.join(tmpdir, '{}.def'.format(basename))
        lname = os.path.join(tmpdir, '{}.lib'.format(basename))

        shutil.copy(pyd, dll)

        definitions = open(dname, 'w+')
        definitions.write('EXPORTS\n')
        dump = iter(
            subprocess.check_output([dumpbin, '/exports', dll], env=env)
            .decode().splitlines())

        while True:
            if next(dump).strip().split() == [
                    'ordinal', 'hint', 'RVA', 'name'
            ]:
                break

        for line in dump:
            if line.strip():
                if line.strip() == 'Summary':
                    break
                else:
                    definitions.write('{}\n'.format(line.strip().split()[-1]))

        definitions.close()
        if 'amd64' in plat_name:
            specifier = '/MACHINE:X64'
        else:
            specifier = '/MACHINE:X86'

        subprocess.call(
            [lib, '/def:{}'.format(dname), '/OUT:{}'.format(lname), specifier],
            env=env)

        yield lname
