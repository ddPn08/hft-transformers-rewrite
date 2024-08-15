import json

from training.config import DatasetConfig


def main():
    with open("./schemas/dataset.schema.json", "w") as f:
        data = DatasetConfig.model_json_schema()
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
