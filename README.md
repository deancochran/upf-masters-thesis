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


# Running the link prediction models:

Inside the root directory run the command:

    python compile.py

By default the data should be downloaded and processed into a **subset**. If you want to you run the full dataset model, 
ajust the arguments outlined in the compile.py file

Once finished continue by running the following command in the same directory:

    python train.py

This will train the three link prediction models needed to preform heterogenous recommendation for a particular user. To adjust the hyper-parameters pass the associated arguments specified inside the train.py file

Additionally, once you run train.py, the graph object will be located in the data/LFM-1b/processed directory ('lastfm1b.bin'). If you wish to train the models on a dataset with different parameters or subset size, delete the 'lastfm1b.bin' dataset file and then run:

    python train.py (-- with all of your args)
    
    
More funcationality will be added soon so that you can make use of the recommendation functions from the terminal. However for now you can use the two jupyter notebooks to understand the link prediction model used to compute node representations.

Finally run the ldm1b-recommendation-test.ipynb notebook to genrerate recommedations utilzing the 3 generated models 
