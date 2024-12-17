import json
import torch
import sys
import os

from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("usage: this.py [dataroot]")
        exit(1)

    dataroot = os.path.abspath(sys.argv[1])
    for stage in Path(dataroot).iterdir():
        index = {}
        for chunk_path in filter(lambda x: x.suffix == ".torch", stage.iterdir()):
            chunk = torch.load(chunk_path)
            for example in chunk:
                index[example["key"]] = str(chunk_path.relative_to(stage))
            with (stage / "index.json").open("w") as writer:
                json.dump(index, writer, indent=4)


if __name__ == "__main__":
    main()
