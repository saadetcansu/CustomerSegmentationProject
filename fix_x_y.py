import json

with open("BIL476_Project.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_list = cell["source"]
        source = "".join(source_list)
        if "X_train, X_test, y_train, y_test = train_test_split(X, y" in source and "One-Hot Encoding" not in source:
            # We need to prepend the definition of X and y
            prep = """# One-Hot Encoding for algorithms
X = pd.get_dummies(df_eng.drop('Response', axis=1), drop_first=True)
y = df_eng['Response']

"""
            new_source = prep + source
            cell["source"] = [line + "\n" if i < len(new_source.split("\n")) - 1 else line for i, line in enumerate(new_source.split("\n"))]

with open("BIL476_Project.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed.")
