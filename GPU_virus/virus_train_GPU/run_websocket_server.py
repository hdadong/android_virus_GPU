import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

import syft as sy
from syft.workers import websocket_server

KEEP_LABELS_DICT = {
    "alice": [0, 1,2,3,4],
    "bob": [0, 1,2,3,4],
    "charlie": [0, 1,2,3,4],
    "testing": list(range(5)),
    #None: list(range(5)),
}
from sklearn import preprocessing

import random
import csv
import pandas
use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
def load_cnn_virus():


    dataframe = pandas.read_csv('0train.csv')


    array = dataframe.values
    #random.shuffle(array) # random the dataset

    features = array[:,0:470]
    labels = array[:,470] - 1


    #print("features222",features.shape)

    #print(len(features[0]))
    features = features.reshape(-1,470) #transfer to a image
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    
    



    features = torch.FloatTensor(features)


    labels = torch.LongTensor(labels)
    
    return (features,labels)
#print(test)




def start_websocket_server_worker(id, host, port, hook, verbose, keep_labels=None, training=True):
    """Helper function for spinning up a websocket server and setting up the local datasets."""
    d = load_cnn_virus()
    server = websocket_server.WebsocketServerWorker(
        id=id, host=host, port=port, hook=hook, verbose=verbose
    )

    if training:

        indices = np.isin(d[1] , keep_labels).astype("uint8")

        logger.info("number of true indices: %s", indices.sum())
        selected_data = (torch.native_masked_select(d[0].transpose(0, 1), torch.tensor(indices)).view(470, -1).transpose(1, 0)).to(device)
        print("selected_data",selected_data)
        #selected_data = d[0]
        logger.info("after selection: %s", selected_data.shape)
        selected_targets = torch.native_masked_select(d[1], torch.tensor(indices)).to(device)
        #selected_targets = d[1]
        dataset = sy.BaseDataset(
            data=selected_data, targets=selected_targets
        )
        key = "mnist"
    else:
        dataset = sy.BaseDataset(
            data=d[0].to(device),
            targets=d[1].to(device),
        )
        key = "mnist_testing"

    server.add_dataset(dataset, key=key)
    count = [0] * 5
    logger.info(
        "MNIST dataset (%s set), available numbers on %s: ", "train" if training else "test", id
    )
    for i in range(5):
        count[i] = (dataset.targets == i).sum().item()
        logger.info("      %s: %s", i, count[i])

    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("len(datasets[mnist]): %s", len(server.datasets[key]))

    server.start()
    return server


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )

    args = parser.parse_args()

    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        keep_labels=KEEP_LABELS_DICT[args.id],
        training=not args.testing,
    )
