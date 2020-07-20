# image_tyle_transform
本项目建立在谷歌大脑和蒙特利尔大学合作的论文 
Exploring the structure of a real-time, 
arbitrary neural artistic stylization network 之上。

## 如何部署
如果你的机器gpu内存足够大，可以把
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
注释掉。

其中mount目录是外部挂载目录，方便dubug。

## 继续训练？
可直接打开我分享的colab地址，将g_model.h5加载后，继续训练。

地址：https://colab.research.google.com/drive/10ts0QiY9urzOQtUBEiTy15Au9DJl7SCc?usp=sharing
