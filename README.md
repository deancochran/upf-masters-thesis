## Environments:
- [Python 3.7](https://pytorch.org/)
- [PyTorch 1.11.0](https://pytorch.org/)
- [Cuda 10.2](https://pytorch.org/)
- [DGL 0.8.2](https://www.dgl.ai/)
- [PyTorch Geometric 2.0.4](https://pytorch-geometric.readthedocs.io/en/latest/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas/pandas)

also don't forget the requirements.txt file


# Running the Recommendation System:

Inside the root directory run the command:

    python compile.py

By default the data should be downloaded and processed into a **subset**. If you want to you run the full dataset model, 
ajust the arguments outlined in the compile.py file (hint: these don't exist yet, hold on!)

Once finished continue by running the following command in the same directory:

    python lastfm1b_train_rhgnn_link_pred.py

This will train the three link prediction models needed to preform heterogenous recommendation for a particular user. 

Finally run the ldm1b-recommendation-test.ipynb notebook to genrerate recommedations utilzing the 3 generated models 
