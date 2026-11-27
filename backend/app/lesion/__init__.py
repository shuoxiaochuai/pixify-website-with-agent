"""眼底病灶分割（CAUNet）。

caunet.py 与 layers.py 逐字取自训练仓库 https://github.com/shuoxiaochuai/DEAR-UNet
（nets/CAUNet.py、nets/layers.py）。权重以 strict=True 加载，模块名与参数名都是
state_dict 的一部分，这两个文件不能为了风格改名或删类。

engine 是唯一的推理入口：预处理、算力判定、阈值化与着色都在那里，
且不在模块顶层 import torch，无卡实例的 app 与 worker 启动不受影响。
"""