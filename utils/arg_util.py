import yaml
import argparse

class YamlArgParser:
    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, yaml_file_path=None):
        if yaml_file_path:
            with open(yaml_file_path, 'r') as yaml_file:
                yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
            args = vars(self.parser.parse_args([]))  # Initialize with default values
            args.update(yaml_data)
            self.args = argparse.Namespace(**args)
        else:
            self.args = self.parser.parse_args()
        return self.args