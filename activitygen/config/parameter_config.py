from configparser import ConfigParser

CONFIG_PARSER = ConfigParser()


def load_config(path):
    CONFIG_PARSER.read(path)
