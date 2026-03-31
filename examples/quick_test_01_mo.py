import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import os
    from matplotlib import pyplot as plt

    return (plt,)


@app.cell
def _():
    a = 'hello'
    a
    return


@app.cell
def _(plt):
    plt.plot([1,2,3,4])
    return


@app.cell
def _(plt):
    import numpy as np
    x = np.linspace(-3, 3, 100)
    y = x ** 2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("U-shaped curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
