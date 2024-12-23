from unitok.utils import Symbols, Symbol


class Status:
    def __init__(self):
        self.status = Symbols.initialized
        # initialized
        # tokenized
        # organized

    @staticmethod
    def require_status(*status: Symbol):
        status_string = '/'.join([s.name for s in status])

        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if self.status in status:
                    return func(self, *args, **kwargs)
                raise ValueError(f'UniTok should be in {status_string} status')

            return wrapper

        return decorator

    require_initialized = require_status(Symbols.initialized)
    require_tokenized = require_status(Symbols.tokenized)
    require_organized = require_status(Symbols.organized)

    require_not_initialized = require_status(Symbols.tokenized, Symbols.organized)
    require_not_tokenized = require_status(Symbols.initialized, Symbols.organized)
    require_not_organized = require_status(Symbols.initialized, Symbols.tokenized)

    @staticmethod
    def change_status(status: Symbol):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                self.status = status
                return result
            return wrapper
        return decorator

    to_tokenized = change_status(Symbols.tokenized)
    to_organized = change_status(Symbols.organized)
