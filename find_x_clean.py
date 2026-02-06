
import json

path = "notebook/03_pca_node_features.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "X_clean =" in source or "X_clean=" in source:
        with open("found_cell.py", "w", encoding="utf-8") as out:
            out.write(source)
        print(f"Cell {i} written to found_cell.py")
        break
