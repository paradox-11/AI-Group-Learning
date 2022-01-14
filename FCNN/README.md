# 全连接网络

## Feature

* `DropOut` 随机舍弃一些单元
* 可自定义的网络结构（目前来说隐藏层全都是用的`relu` / `leaky relu`，然后输出层是`sigmoid` / `softMax`，更换激活函数需要手动改代码）
* `batch` 分批处理
