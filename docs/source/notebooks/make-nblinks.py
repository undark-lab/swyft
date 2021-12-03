import json
from pathlib import Path


def create_nblink(notebook_path):
    d = {}
    d["path"] = str(notebook_path)
    d["extra-media"] = []
    return d


def main():
    root = Path(__file__).parent
    relative_source = Path("../../../notebooks/")
    source = Path(root, relative_source)
    for nbpath in source.glob("*/"):
        if ".ipynb_checkpoints" in str(nbpath):
            continue
        else:
            target_path = Path(relative_source, f"{nbpath.stem}.ipynb")
            nblink = create_nblink(target_path)
            linkpath = Path(root, f"{nbpath.stem}.nblink")
            with open(linkpath, "w") as f:
                print("Creating path for", target_path, "Located at", linkpath)
                json.dump(nblink, f)


if __name__ == "__main__":
    main()
