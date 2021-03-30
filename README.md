# 使用说明
<br>DNN文件夹下为可执行代码
<br>服务器共分为协调端、测试端、训练端
<br>协调端在本地创建一个虚拟worker，接受远程实际worker上传的训练模型参数，并发送给远程实际的测试woker进行测试
<br>在具体实现中，我们把协调端和测试端放在同一个服务器上面。

# 环境要求
<br>极客云上租的服务器
<br>python3.7
<br>pytorch-1.4
<br>PySyft-0.2.4版本
# 环境安装
```
# 添加清华源的pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
# conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch #下载CPU版本pytorch
pip3 install syft==0.2.4 --no-dependencies
# 安装以下依赖
pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 
pip install websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 
pip install requests~=2.22.0 numpy~=1.18.1
```

# 代码执行
## 测试段/协调端服务器执行代码
```
git clone git@github.com:hdadong/android_virus_GPU.git
cd android_virus_GPU
git checkout dnn
cd DNN
cd virus_test_GPU
ll
```
    文件目录如下：
    test.csv :1000个APP的470维特征+标签
    run_websocket_client.py ：修改模型的参数和训练epoch,学习率，优化器、batchsize等参数
    run_websocket_server_testing.py：对test.csv数据进行归一化，读入数据到内存，挂起测试woker，等待测试。
    Asynchronous-federated-learning-on-MNIST.ipynb：这是协调端的代码
```
nohup  python -u run_websocket_server_testing.py --host 0.0.0.0  --port 28007 --id testing > test.out 2>&1 &  
```
<br>后台挂起test woker，这里的IP和端口需要修改，执行完这句代码后，测试woker将会被挂起，等待协调端把要测试的模型参数、模型结构、训练参数发送过来，发送过来后才会被唤醒进行测试。测试结束后会把结果返回给协调端。

```
cat test.out 查看输出
```
```
设置过滤标签
KEEP_LABELS_DICT = {
    "alice": list(range(15)),
    "bob": list(range(15)),
    "charlie": list(range(15)),
    "testing": list(range(15)),
    #None: list(range(5)),
}
```
## 训练端服务器执行代码
```
git clone git@github.com:hdadong/android_virus_GPU.git
cd android_virus_GPU
git checkout dnn
cd DNN
cd virus_train_GPU  
ll  
```
    文件目录：  
    0train.csv:5000个APP的470维特征+标签  
    run_websocket_server.py：读入文件夹下0train.csv并做归一化，挂起train woker。  
```
nohup  python -u run_websocket_server.py --host 0.0.0.0  --port 28068 --id bob > test.out 2>&1 &   
```
<br>这里的IP、端口需、woker的名字（alice或者bob)要修改，执行完这句代码后，测试woker将会被挂起，等待协调端把模型参数、模型结构、训练参数发送过来，被唤醒后会调用本地数据集进行训练，训练结束后会把模型参数发送回给协调端  
```
设置过滤标签
KEEP_LABELS_DICT = {
    "alice": list(range(15)),
    "bob": list(range(15)),
    "charlie": list(range(15)),
    "testing": list(range(15)),
    #None: list(range(5)),
}
```
## 协调端  
    按顺序执行Asynchronous-federated-learning-on-MNIST.ipynb的代码段  
    需要注意的是 
### clien.py
```
def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=1000, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.8, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
```    
### 第六个代码段  
![image](https://user-images.githubusercontent.com/44421595/112457567-db5f5180-8d96-11eb-8705-4f04452fadd8.png)  
    需要修改训练和测试woker的IP和端口  
### 第九个代码段：  
```
traced_model = torch.jit.trace(model, torch.zeros([1, 1, 470], dtype=torch.float).to(device))
```
    470为模型的输入维度，这行代码是把模型结构和参数进行序列化。
### 第十一个代码段
    若协调端没有gpu则要把device = "cuda"  改为 "cpu"   
```
traced_model = torch.jit.trace(model, torch.zeros([1, 1, 470], dtype=torch.float).to(device))  
```
    470为模型的输入维度，这行代码是把模型结构和参数进行序列化后再传送  
    await asyncio.gather 是等待所有的训练节点返回模型的参数后再继续训练  
    learning_rate = max(0.99 * learning_rate, args.lr * 0.01)  设置lr的衰减率  
```
            rwc.fit_model_on_worker(
                worker=worker,
                traced_model=traced_model,
                batch_size=args.batch_size,
                curr_round=curr_round,
                max_nr_batches=args.federate_after_n_batches, #当epoch =1时，若max_nr_batches>0,则本地服务器训练了若max_nr_batches个batch就会停止训练。当epoch >1时，若max_nr_batches>0,则本地服务器每个epoch只训练了max_nr_batches个batch。
                所以这个参数应该置为负数。
                lr=learning_rate,
            )
            
                        rwc.evaluate_model_on_worker(
                model_identifier="Model update " + worker_id,
                worker=testing,
                dataset_key="mnist_testing",
                model=worker_model,
                nr_bins=5,
                batch_size=128,
                print_target_hist=False,
                device=device
            )
```

# 参考文献
环境配置参考  
http://www.shangdixinxi.com/detail-1590606.html  
pysyft使用参考  
1.https://zhuanlan.zhihu.com/p/181733116?utm_source=wechat_session  
2.https://zhuanlan.zhihu.com/p/349204625



