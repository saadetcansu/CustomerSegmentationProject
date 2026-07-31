import json

with open("BIL476_Project.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_list = cell["source"]
        source = "".join(source_list)
        if "probabilities = best_pipe.predict_proba(X_selected)[:, 1]" in source:
            new_source = source.replace("X_selected", "pd.concat([X_train_selected, X_test_selected]).sort_index()")
            cell["source"] = [line + "\n" if i < len(new_source.split("\n")) - 1 else line for i, line in enumerate(new_source.split("\n"))]

with open("BIL476_Project.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed.")
