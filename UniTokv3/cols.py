class Cols(dict):
    def get_info(self):
        return {col.name: col.get_info() for col in self.values()}
