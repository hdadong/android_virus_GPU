＃在GPU下训练CNN

使用0train.csv训练
修改了client中的读入预处理，进行了Minmax归一化，之前master的client的代码有问题

![image](https://user-images.githubusercontent.com/44421595/111877777-ddfd2800-89df-11eb-806c-2a9e3606cc31.png)
lr=1 batchsize 512 68%


![image](https://user-images.githubusercontent.com/44421595/111877788-ea818080-89df-11eb-8a43-46977ed2819b.png)
74%  lr=1 batchsize 512

![image](https://user-images.githubusercontent.com/44421595/111877812-05ec8b80-89e0-11eb-91be-46063a1848dd.png)
70%  lr=1 batchsize 512


![image](https://user-images.githubusercontent.com/44421595/111877822-0b49d600-89e0-11eb-8ce8-7833aff300af.png)
73%  lr=1 batchsize 512
