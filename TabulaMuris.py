# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import scanpy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    anndata = scanpy.read_h5ad("/home/eluator/Downloads/tabula-muris-senis-facs-official-raw-obj.h5ad")
    print(anndata.X)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
