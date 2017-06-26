import string

from setuptools import Extension


class WinExt(Extension):
    # Base class for all win32 extensions, with some predefined
    # library and include dirs, and predefined windows libraries.
    # Additionally a method to parse .def files into lists of exported
    # symbols, and to read

    def __init__(
            self,
            name,
            sources,
            include_dirs=[],
            define_macros=None,
            undef_macros=None,
            library_dirs=[],
            libraries=[],
            runtime_library_dirs=None,
            extra_objects=None,
            extra_compile_args=None,
            extra_link_args=None,
            export_symbols=None,
            export_symbol_file=None,
            pch_header=None,
            windows_h_version=None,  # min version of windows.h needed.
            extra_swig_commands=None,
            is_regular_dll=False,  # regular Windows DLL?
            # list of headers which may not be installed forcing us to
            # skip this extension
            optional_headers=[],
            depends=None,
            platforms=None,  # none means 'all platforms'
            unicode_mode=None,
            # 'none'==default or specifically true/false.
            implib_name=None,
            delay_load_libraries='',
            external_modules=[],  # Libs that needed for building
            build_temp_library_dirs=[]):

        self.delay_load_libraries = delay_load_libraries.split()
        libraries.extend(self.delay_load_libraries)

        if export_symbol_file:
            export_symbols = export_symbols or []
            export_symbols.extend(self.parse_def_file(export_symbol_file))

        # Some of our swigged files behave differently in distutils vs
        # MSVC based builds.  Always define DISTUTILS_BUILD so they can tell.
        define_macros = define_macros or []
        define_macros.append(('DISTUTILS_BUILD', None))
        define_macros.append(('_CRT_SECURE_NO_WARNINGS', None))

        self.pch_header = pch_header
        self.extra_swig_commands = extra_swig_commands or []
        self.windows_h_version = windows_h_version
        self.optional_headers = optional_headers
        self.is_regular_dll = is_regular_dll
        self.platforms = platforms
        self.implib_name = implib_name
        self.external_modules = external_modules
        self.build_temp_library_dirs = build_temp_library_dirs
        Extension.__init__(self, name, sources, include_dirs, define_macros,
                           undef_macros, library_dirs, libraries,
                           runtime_library_dirs, extra_objects,
                           extra_compile_args, extra_link_args, export_symbols)

        if not hasattr(self, 'swig_deps'):
            self.swig_deps = []
        self.extra_compile_args.extend(['/DUNICODE', '/D_UNICODE', '/DWINNT'])
        self.unicode_mode = unicode_mode

        if self.delay_load_libraries:
            self.libraries.append('delayimp')
            for delay_lib in self.delay_load_libraries:
                self.extra_link_args.append('/delayload:%s.dll' % delay_lib)

        if not hasattr(self, '_needs_stub'):
            self._needs_stub = False

    def parse_def_file(self, path):
        # Extract symbols to export from a def-file
        result = []
        for line in open(path).readlines():
            line = line.rstrip()
            if line and line[0] in string.whitespace:
                tokens = line.split()
                if not tokens[0][0] in string.ascii_letters:
                    continue
                result.append(','.join(tokens))
        return result
