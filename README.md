#使用说明
DNN文件夹下为可执行代码
服务器共分为协调端、测试端、训练端
协调端在本地创建一个虚拟worker，接受远程实际worker上传的训练模型参数，并发送给远程实际的测试woker进行测试
在具体实现中，我们把协调端和测试端放在同一个服务器上面。

#代码执行
##测试段/协调端服务器执行代码
git clone git@github.com:hdadong/android_virus_GPU.git
cd android_virus_GPU
git checkout dnn
cd DNN
cd virus_test_GPU
ll
文件目录如下：
test.csv :1000个APP的470维特征+标签
run_websocket_client.py ：修改模型的参数和训练epoch,学习率，优化器、batchsize等参数
run_websocket_server_testing.py：对test.csv数据进行归一化，读入数据到内存，挂起测试woker，等待测试。
Asynchronous-federated-learning-on-MNIST.ipynb：这是协调端的代码

nohup  python -u run_websocket_server_testing.py --host 0.0.0.0  --port 28007 --id testing > test.out 2>&1 &  
后台挂起test woker，这里的IP和端口需要修改，执行完这句代码后，测试woker将会被挂起，等待协调端把要测试的模型参数、模型结构、训练参数发送过来，发送过来后才会被唤醒进行测试。测试结束后会把结果返回给协调端。
cat test.out 查看输出


##训练端服务器执行代码
git clone git@github.com:hdadong/android_virus_GPU.git
cd android_virus_GPU
git checkout dnn
cd DNN
cd virus_train_GPU
ll
文件目录：
0train.csv:5000个APP的470维特征+标签
run_websocket_server.py：读入文件夹下0train.csv并做归一化，挂起train woker。

nohup  python -u run_websocket_server.py --host 0.0.0.0  --port 28068 --id bob > test.out 2>&1 & 
这里的IP、端口需、woker的名字（alice或者bob)要修改，执行完这句代码后，测试woker将会被挂起，等待协调端把模型参数、模型结构、训练参数发送过来，被唤醒后会调用本地数据集进行训练，训练结束后会把模型参数发送回给协调端

##协调端
按顺序执行Asynchronous-federated-learning-on-MNIST.ipynb的代码段
需要注意的是
###第六个代码段
![image](https://user-images.githubusercontent.com/44421595/112457567-db5f5180-8d96-11eb-8705-4f04452fadd8.png)
需要修改训练和测试woker的IP和端口
###第九个代码段：
traced_model = torch.jit.trace(model, torch.zeros([1, 1, 470], dtype=torch.float).to(device))
470为模型的输入维度，这行代码是把模型结构和参数进行序列化。
###第十一个代码段
若协调端没有gpu则要把device = "cuda"  改为 "cpu" 
traced_model = torch.jit.trace(model, torch.zeros([1, 1, 470], dtype=torch.float).to(device))
470为模型的输入维度，这行代码是把模型结构和参数进行序列化后再传送
await asyncio.gather 是等待所有的训练节点返回模型的参数后再继续训练
learning_rate = max(0.99 * learning_rate, args.lr * 0.01)
设置lr的衰减率


#参考文献
环境配置参考
http://www.shangdixinxi.com/detail-1590606.html
pysyft使用参考
1.https://zhuanlan.zhihu.com/p/181733116?utm_source=wechat_session
2.https://zhuanlan.zhihu.com/p/349204625
