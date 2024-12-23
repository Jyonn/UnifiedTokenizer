from pigmento import Pigmento, Prefix, Color, DynamicColorPlugin, Bracket


class Verbose:
    DEBUG = 3
    INFO = 2
    WARNING = 1
    ERROR = 0

    level = INFO

    @classmethod
    def set_level(cls, level):
        cls.level = level

    @classmethod
    def get_printer(cls, level):
        def printer(prefixes, prefix_s, prefix_s_with_color, text, **kwargs):
            if level <= cls.level:
                print(prefix_s_with_color, text, **kwargs)
        return printer


debug = Pigmento()
debug.add_prefix(Prefix('D', bracket=Bracket.DEFAULT, color=Color.GREEN))
debug.set_basic_printer(Verbose.get_printer(Verbose.DEBUG))

info = Pigmento()
info.add_prefix(Prefix('I', bracket=Bracket.DEFAULT, color=Color.BLUE))
info.set_basic_printer(Verbose.get_printer(Verbose.INFO))

warning = Pigmento()
warning.add_prefix(Prefix('W', bracket=Bracket.DEFAULT, color=Color.YELLOW))
warning.set_basic_printer(Verbose.get_printer(Verbose.WARNING))

error = Pigmento()
error.add_prefix(Prefix('E', bracket=Bracket.DEFAULT, color=Color.RED))
error.set_basic_printer(Verbose.get_printer(Verbose.ERROR))

for pnt in [debug, info, warning, error]:
    pnt.add_plugin(DynamicColorPlugin(
        Color.MAGENTA, Color.BLUE, Color.RED, Color.YELLOW, Color.GREEN, Color.CYAN))
    pnt.set_display_mode(
        display_method_name=False,
        display_class_name=True
    )

Verbose.set_level(Verbose.WARNING)
