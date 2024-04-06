import nbformat as nbf
import glob

def py_to_nb(py_file):
    with open(py_file) as f:
        code = f.read()

    return nbf.v4.new_code_cell(source=code)

py_files = glob.glob('*.py')

nb = nbf.v4.new_notebook()

for py_file in py_files:
    new_cell = py_to_nb(py_file)
    nb.cells.append(new_cell)

with open('combined.ipynb', 'w') as f:
    nbf.write(nb, f)