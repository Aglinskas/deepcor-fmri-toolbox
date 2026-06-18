# Agent instructions

## Working with Jupyter notebooks (.ipynb)

Do **not** read or edit `.ipynb` files directly unless absolutely necessary.
`.ipynb` files are large JSON blobs that often embed rendered plots, base64
images, and other output that is noise for code inspection and wastes context.

Instead, first convert the notebook to a plain Python file and inspect that:

```bash
jupyter nbconvert notebook.ipynb --to python
```

Read the generated `notebook.py` to understand the code. Only fall back to
operating on the `.ipynb` directly when a change must land in the notebook
itself and cannot be made any other way.

## Keep README.md and the wiki in sync with the code

`README.md` documents the package structure and lists the functions/classes in
each module (with GitHub links to the files). The `wiki/` folder holds
user-facing guides (data preparation, the high-level API, etc.) that describe
parameters, defaults, and example usage. Whenever you edit code in a way that
changes the public surface, check whether the README **and** any relevant
`wiki/` page need updating, and update them if so. In particular, after you:

- rename, add, or remove a function, method, or class,
- add or remove a file/module,
- change a function signature shown in a README example (argument names, etc.),
- or change install steps, dependencies, or example usage,

re-read the relevant part of `README.md` (and any affected `wiki/` page) and
bring it back in line with the code. Don't leave the docs describing functions,
files, parameters, or defaults that no longer match what's in the package.
