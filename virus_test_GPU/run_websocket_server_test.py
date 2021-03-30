import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

import syft as sy
from syft.workers import websocket_server
from sklearn import preprocessing

KEEP_LABELS_DICT = {
    "alice": [0, 1,2,3,4],
    "bob": [0, 1,2,3,4],
    "charlie": [0, 1,2,3,4],
    "testing": list(range(5)),
    None: list(range(5)),
}
use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")
import random
import csv
import pandas
def load_cnn_virus():


    dataframe = pandas.read_csv('test.csv')
    #dataframe1 = dataframe[['FS_ACCESS(CREATE__APPEND)__','NETWORK_ACCESS(READ__WRITE__)','NETWORK_ACCESS(WRITE__)__','add','addAccountExplicitly','addPeriodicSync','addStatusChangeListener','addWithoutInputChannel','attachEngine','cancelAllNotifications','cancelSync','cancelToast','cancelVibrate','clock_getres','collapsePanels','deleteHost','engineShown','faccessat','fchmod','fcntl','finishInput','fstatfs64','ftruncate64','geocoderIsPresent','getAccounts','getAllCellInfo','getAllPkgUsageStats','getAppWidgetInfo','getApplicationRestrictions','getAuthenticatorTypes','getBoolean','getCameraDisabled','getDataActivity','getDeviceId','getDeviceList','getDeviceSvn','getDhcpInfo','getDisplayIds','getEnabledInputMethodSubtypeList','getFlashlightEnabled','getInputDevice','getInstalledProviders','getIsSyncable','getLastChosenActivity','getLastInputMethodSubtype','getLong','getMessenger','getMobileIfaces','getNetworkPreference','getNightMode','getNumberOfCameras','getPackageGids','getPermissionGroupInfo','getPrimaryClip','getProviderProperties','getReceiverInfo','getSearchableInfo','getState','getStorageEncryptionStatus','getString','getSyncAutomatically','getSystemAvailableFeatures','getTetheredIfaces','getUsers','getVoiceMailAlphaTag','getVoiceMailNumber','getWifiDisplayStatus','getresgid32','hasIccCard','hasKeys','hasPrimaryClip','hideSoftInput','isActiveNetworkMetered','isBluetoothA2dpOn','isBluetoothScoOn','isImsSmsSupported','isKeyguardSecure','isPackageAvailable','isSafeMode','isSpeakerphoneOn','isSyncActive','mount','munlock','onFinished','partiallyUpdateAppWidgetIds','pingSupplicant','play','playSoundEffect','playSoundEffectVolume','prepareVpn','queryIntentContentProviders','queryIntentReceivers','reassociate','reenableKeyguard','registerMediaButtonIntent','registerRemoteControlClient','registerSuggestionSpansForNotification','releaseMulticastLock','removeAccessibilityInteractionConnection','removeActiveAdmin','removePrimaryClipChangedListener','removeStatusChangeListener','requestScanFile','resolveContentProvider','resolveService','rt_sigprocmask','sched_get_priority_max','sched_setaffinity','sendExtraCommand','sendMultipartText','setApplicationEnabledSetting','setCallback','setComponentEnabledSetting','setDiscoveryRequest','setExtractedText','setFlashlightEnabled','setIsSyncable','setNetworkPreference','setPlaybackStateForRcc','setPrimaryClip','setRadio','setSpeakerphoneOn','setStreamVolume','setSyncAutomatically','setWallpaper','setWifiApConfiguration','setWifiEnabled','showSoftInput','showStrictModeViolation','startBluetoothSco','startWatchingRoutes','startWifiDisplayScan','stopListening','stopUsingNetworkFeature','symlink','timer_create','toggleSoftInput','truncate','uname','unregisterMediaButtonIntent','unregisterRemoteControlClient','updateAppWidgetProvider','vibratePattern','watchRotation']]

    array = dataframe.values
    #random.shuffle(array) # random the dataset
    features = array[:,0:470]
    #features = dataframe1.values
    labels = array[:,470] - 1
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    
    features = torch.FloatTensor(features)
    #print("features222",features.shape)

    #print(len(features[0]))
    features = features.reshape(-1,470) #transfer to a image
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
        #print(d[0].shape)
        #print(mnist_dataset.data.transpose(0, 2).shape)
        indices = np.isin(d[1] , keep_labels).astype("uint8")
        #print((torch.tensor(indices)).shape)
        logger.info("number of true indices: %s", indices.sum())
        selected_data = (
            torch.native_masked_select(d[0].transpose(0, 1), torch.tensor(indices))
            .view(470, -1)
            .transpose(1, 0)
        )
        print(selected_data)
        logger.info("after selection: %s", selected_data.shape)
        selected_targets = torch.native_masked_select(d[1], torch.tensor(indices))

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
        training= args.testing,
    )
