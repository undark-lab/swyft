import json
from pathlib import Path

IGNORE = [
    ".ipynb_checkpoints",
    "Video",
]


def create_nblink(notebook_path):
    d = {}
    d["path"] = str(notebook_path)
    d["extra-media"] = []
    return d


def main():
    root = Path(__file__).parent
    # clean directory first
    for old_nblink in root.glob("*.nblink"):
        Path(old_nblink).unlink()

    # add the relevant notebooks and names
    relative_source = Path("../../../notebooks/")
    source = Path(root, relative_source)
    for nbpath in source.glob("*/"):
        if any([ign in str(nbpath) for ign in IGNORE]):
            continue
        elif nbpath.is_dir():
            continue
        elif nbpath.suffix != ".ipynb":
            continue
        else:
            target_path = Path(relative_source, f"{nbpath.stem}.ipynb")
            nblink = create_nblink(target_path)
            linkpath = Path(root, f"{nbpath.stem}.nblink")
            with open(linkpath, "w") as f:
                print("Creating path for", target_path, "Located at", linkpath)
                json.dump(nblink, f)
                f.writelines("\n")


if __name__ == "__main__":
    main()
