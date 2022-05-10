import hydra
from omegaconf import DictConfig

from process import process_data
from feature_engineer import feature_data


@hydra.main(config_path="../config", config_name="main")
def main(config: DictConfig):

    process_data(config)
    feature_data(config)


if __name__ == "__main__":
    main()