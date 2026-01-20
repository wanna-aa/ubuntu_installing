# Ubuntu 22.04 NVIDIA 深度学习环境配置指南

> Ubuntu 22.04 LTS + NVIDIA Driver + CUDA 11.8 + cuDNN 完整安装教程

[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 目录

- [环境说明](#环境说明)
- [安装前检查](#0-安装前检查)
- [安装 NVIDIA 驱动](#1-安装-nvidia-显卡驱动)
- [安装 CUDA 11.8](#2-安装-cuda-118)
- [安装 cuDNN](#3-安装-cudnn)
- [验证安装](#4-验证-cudnn-是否安装成功)
- [深度学习框架验证](#5-可选pytorch--tensorflow-简单验证)
- [常见问题](#6-常见问题排查)
- [卸载指南](#7-卸载需要时)

## 🎯 环境说明

**目标配置：**
- **操作系统**：Ubuntu 22.04 LTS
- **NVIDIA 驱动**：推荐 535+ (根据系统推荐)
- **CUDA 版本**：11.8
- **cuDNN 版本**：8.x (兼容 CUDA 11.x)

**适用场景：**
- 深度学习模型训练 (PyTorch, TensorFlow)
- CUDA 开发与测试
- GPU 加速计算

---

## 0. 安装前检查

### 0.1 确认显卡型号
```bash
lspci | grep -i nvidia
```

**预期输出示例：**
```
01:00.0 VGA compatible controller: NVIDIA Corporation ...
```

### 0.2 更新系统（推荐）
```bash
sudo apt update
sudo apt -y upgrade
sudo reboot
```

### 0.3 ⚠️ 重要：Secure Boot 问题

**问题表现：** 安装驱动后 `nvidia-smi` 失败，提示无法加载内核模块

**解决方案（二选一）：**

1. **推荐方式**：进入 BIOS 关闭 Secure Boot
   - 重启时按 F2/F10/Del 进入 BIOS
   - 找到 Security → Secure Boot → Disabled
   - 保存并重启

2. **高级方式**：安装驱动后进行 MOK (Machine Owner Key) 注册
```bash
   sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
   # 按提示设置密码，重启后在蓝屏界面完成注册
```

---

## 1. 安装 NVIDIA 显卡驱动

### 1.1 清理旧安装（重要）
```bash
# 卸载所有旧版本
sudo apt -y purge 'nvidia*' 'cuda*' 'libcudnn*'
sudo apt -y autoremove

# 清理残留配置
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600

# 重启系统
sudo reboot
```

### 1.2 查看推荐驱动并安装
```bash
# 查看系统推荐的驱动版本
sudo ubuntu-drivers devices
```

**输出示例：**
```
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
...
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-530 - distro non-free
```

**安装推荐版本：**
```bash
sudo apt update
sudo apt -y install nvidia-driver-535  # 替换为你的推荐版本
sudo reboot
```

### 1.3 验证驱动安装
```bash
nvidia-smi
```

**成功输出示例：**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx      Driver Version: 535.xxx      CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
...
```

> 💡 **注意**：`nvidia-smi` 显示的 CUDA Version 是驱动支持的最高版本，不是实际安装的 CUDA Toolkit 版本

---

## 2. 安装 CUDA 11.8

### 2.1 安装必要依赖
```bash
sudo apt update
sudo apt -y install build-essential dkms linux-headers-$(uname -r) wget gnupg
```

### 2.2 添加 CUDA 11.8 官方软件源
```bash
# 下载并安装 CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 更新软件包列表
sudo apt update
```

### 2.3 安装 CUDA 11.8 Toolkit
```bash
sudo apt -y install cuda-toolkit-11-8
```

**安装完成后重启：**
```bash
sudo reboot
```

### 2.4 配置环境变量

**方法一：写入 `~/.bashrc`（推荐）**
```bash
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**方法二：系统全局配置（可选）**
```bash
sudo tee /etc/profile.d/cuda.sh > /dev/null <<'EOF'
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

source /etc/profile.d/cuda.sh
```

### 2.5 验证 CUDA 安装
```bash
nvcc -V
```

**预期输出：**
```
nvcc: NVIDIA (R) Cuda compiler driver
...
Cuda compilation tools, release 11.8, V11.8.xxx
```

**（可选）编译 CUDA Samples 验证：**
```bash
# 克隆 CUDA Samples（如果系统未自带）
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
make

# 运行测试
./deviceQuery
```

---

## 3. 安装 cuDNN

> 📥 **下载地址**：[NVIDIA cuDNN Download](https://developer.nvidia.com/cudnn) (需要注册登录)

**选择版本：** cuDNN 8.x for CUDA 11.x

### 方式 A：使用 .deb 安装（推荐）

#### A1. 使用本地仓库包

如果下载的是 `cudnn-local-repo-ubuntu2204-8.x.x.x_1.0-1_amd64.deb`：
```bash
# 安装本地仓库
sudo dpkg -i cudnn-local-repo-ubuntu2204-*.deb

# 复制 GPG 密钥
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/

# 更新并安装
sudo apt update
sudo apt -y install libcudnn8 libcudnn8-dev
```

#### A2. 使用分包安装

如果下载的是多个 `.deb` 文件（`libcudnn8_*.deb`, `libcudnn8-dev_*.deb`）：
```bash
sudo dpkg -i libcudnn8_*.deb
sudo dpkg -i libcudnn8-dev_*.deb

# 可选：安装示例代码
sudo dpkg -i libcudnn8-samples_*.deb
```

### 方式 B：使用 tar 包安装（通用）

如果下载的是 `cudnn-linux-x86_64-8.x.x.x_cuda11-archive.tar.xz`：
```bash
# 解压
tar -xf cudnn-linux-x86_64-*_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-*_cuda11-archive

# 复制文件到 CUDA 目录
sudo cp -P include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp -P lib/libcudnn* /usr/local/cuda-11.8/lib64/

# 设置权限
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

# 更新链接库缓存
sudo ldconfig
```

---

## 4. 验证 cuDNN 是否安装成功

### 4.1 检查库文件链接
```bash
ldconfig -p | grep cudnn
```

**预期输出：**
```
libcudnn.so.8 (libc6,x86-64) => /usr/local/cuda-11.8/lib64/libcudnn.so.8
libcudnn_cnn_infer.so.8 (libc6,x86-64) => /usr/local/cuda-11.8/lib64/...
```

### 4.2 检查 cuDNN 版本
```bash
grep -A 2 'CUDNN_MAJOR' /usr/local/cuda-11.8/include/cudnn_version.h
```

**预期输出：**
```
#define CUDNN_MAJOR 8
#define CUDNN_MINOR x
#define CUDNN_PATCHLEVEL x
```

---

## 5. （可选）PyTorch / TensorFlow 简单验证

### 5.1 PyTorch 验证

**安装 PyTorch (CUDA 11.8)：**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**验证脚本：**
```bash
python3 - << 'PY'
import torch
print("=" * 50)
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 设备:", torch.cuda.get_device_name(0))
    print("CUDA 运行时版本:", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("=" * 50)
    # 简单测试
    x = torch.rand(5, 3).cuda()
    print("✅ GPU 张量计算测试通过")
else:
    print("❌ CUDA 不可用，请检查配置")
print("=" * 50)
PY
```

### 5.2 TensorFlow 验证

**安装 TensorFlow：**
```bash
pip3 install tensorflow[and-cuda]
```

**验证脚本：**
```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

---

## 6. 常见问题排查

### Q1：`nvidia-smi` 报错或显示 "No devices were found"

**可能原因：**
1. ✗ Secure Boot 未关闭
2. ✗ 驱动模块未正确加载
3. ✗ 内核头文件/DKMS 编译问题

**排查步骤：**
```bash
# 检查内核日志
sudo dmesg | grep -i nvidia | tail -n 50

# 检查模块加载状态
lsmod | grep nvidia

# 重新编译驱动模块（如有必要）
sudo dkms autoinstall
sudo modprobe nvidia
```

### Q2：`nvcc` 命令找不到

**解决方法：**
```bash
# 检查 nvcc 是否存在
ls -l /usr/local/cuda-11.8/bin/nvcc

# 检查 PATH 配置
echo $PATH | grep cuda

# 重新加载环境变量
source ~/.bashrc
```

### Q3：运行时报错 "libcudnn.so: cannot open shared object file"

**解决方法：**
```bash
# 检查 LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# 检查 cuDNN 库文件
ls -l /usr/local/cuda-11.8/lib64/libcudnn*

# 更新库缓存
sudo ldconfig

# 验证库链接
ldconfig -p | grep cudnn
```

### Q4：驱动和 CUDA 版本如何搭配？

**版本兼容原则：**

| CUDA Toolkit | 最低驱动版本 (Linux) | 推荐驱动版本 |
|--------------|---------------------|-------------|
| 11.8         | ≥ 520.61.05         | 535+        |
| 12.0         | ≥ 525.60.13         | 535+        |

> 💡 **关键点**：
> - 驱动版本 **向后兼容** CUDA Runtime（新驱动支持旧 CUDA）
> - 深度学习框架需要匹配对应的 CUDA/cuDNN 版本
> - 使用 `nvidia-smi` 查看驱动版本，`nvcc -V` 查看 CUDA Toolkit 版本

### Q5：多 CUDA 版本共存

**场景：** 需要同时使用 CUDA 11.8 和 12.x
```bash
# 创建软链接切换版本
sudo ln -sf /usr/local/cuda-11.8 /usr/local/cuda

# 或在 ~/.bashrc 中动态设置
export CUDA_HOME=/usr/local/cuda-11.8  # 修改此处切换版本
```

---

## 7. 卸载（需要时）

### 7.1 卸载 CUDA Toolkit
```bash
sudo apt -y remove --purge cuda-toolkit-11-8
sudo apt -y autoremove

# 清理残留文件
sudo rm -rf /usr/local/cuda-11.8
```

### 7.2 卸载 cuDNN
```bash
# .deb 方式安装的
sudo apt -y remove --purge libcudnn8*

# tar 方式安装的
sudo rm -f /usr/local/cuda-11.8/include/cudnn*.h
sudo rm -f /usr/local/cuda-11.8/lib64/libcudnn*
sudo ldconfig
```

### 7.3 卸载 NVIDIA 驱动
```bash
sudo apt -y purge 'nvidia*'
sudo apt -y autoremove
sudo reboot
```

---

## 📚 参考资料

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
