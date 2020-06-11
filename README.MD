# 依赖
你可以使用pip install 安装所有的依赖
 * Python >= 3.5
 * Pytorch >= 0.4.0
 * ignite = 0.1.2 
 * torchvision
 * yacs

# 运行相关
 * 模型定义: modeling/
 * loss函数: loss/\_init_.py
 * 训练过程: engine/trainer.py
 
# 训练
## 开始训练
参考命令 sh train_base_alltricks.sh

## 参数设置
参考configs/下的.yml文件