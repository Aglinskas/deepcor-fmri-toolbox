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
