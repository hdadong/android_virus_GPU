import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

import syft as sy
from syft.workers import websocket_server

KEEP_LABELS_DICT = {
    "alice": list(range(15)),
    "bob": list(range(15)),
    "charlie": list(range(15)),
    "testing": list(range(15)),
    #None: list(range(5)),
}
from sklearn import preprocessing

import random
import csv
import pandas
use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


import pandas as pd
def start_websocket_server_worker(id, host, port, hook, verbose, keep_labels=None, training=True):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = websocket_server.WebsocketServerWorker(
        id=id, host=host, port=port, hook=hook, verbose=verbose
    )

    dataset = pd.read_csv("0.csv",dtype=float, iterator=True)
    key = "mnist"
    server.add_dataset(dataset, key=key)
    dataset = None
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
