import click
import dotenv

from .config import UnetConfig
from .train import start_training


@click.command()
@click.argument("config", type=click.Path(exists=True))
def main(config):
    dotenv.load_dotenv("../.env")
    config = UnetConfig.from_json(config)
    start_training(config)


if __name__ == "__main__":
    main()
