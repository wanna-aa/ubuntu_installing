Ubuntu 22.04 + NVIDIA Driver + CUDA 11.8 + cuDNN 安装指南

目标环境：Ubuntu 22.04 LTS
目标组件：NVIDIA 显卡驱动 + CUDA 11.8 + cuDNN（与 CUDA 11.8 兼容的版本）

0. 安装前检查
0.1 确认显卡型号
lspci | grep -i nvidia

0.2 （可选）更新系统
sudo apt update
sudo apt -y upgrade
sudo reboot

0.3 常见坑：Secure Boot

如果你启用了 Secure Boot，NVIDIA 驱动模块可能无法加载（表现为 nvidia-smi 失败）。

推荐：进 BIOS 关闭 Secure Boot

或：安装驱动后自行进行 MOK enrol（较麻烦，本文不展开）

1. 安装 NVIDIA 显卡驱动（推荐使用 Ubuntu 官方仓库）
1.1 清理可能存在的旧安装
sudo apt -y purge 'nvidia*' 'cuda*' 'libcudnn*'
sudo apt -y autoremove
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600
sudo reboot

1.2 查看系统推荐驱动版本并安装
sudo ubuntu-drivers devices


安装推荐版本（示例：如果推荐为 nvidia-driver-535，就执行下面命令）：

sudo apt update
sudo apt -y install nvidia-driver-535
sudo reboot

1.3 验证驱动是否正常
nvidia-smi


能看到 GPU 信息和驱动版本即 OK。

2. 安装 CUDA 11.8（使用 NVIDIA 官方 CUDA Repo）

注意：CUDA 可以通过 runfile 安装，但更推荐 APT Repo，便于管理与卸载。

2.1 安装基础依赖
sudo apt update
sudo apt -y install build-essential dkms linux-headers-$(uname -r) wget gnupg

2.2 添加 CUDA 11.8 软件源（Ubuntu 22.04）
# 下载并安装 cuda-keyring（用于添加 NVIDIA 源的 GPG key）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update

2.3 安装 CUDA 11.8 Toolkit
sudo apt -y install cuda-toolkit-11-8


安装完成后建议重启：

sudo reboot

2.4 配置环境变量（推荐写入 ~/.bashrc）
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

2.5 验证 CUDA 安装
nvcc -V


期望输出包含 release 11.8。

（可选）进一步检查 CUDA Samples：

# Ubuntu repo 安装的 samples 可能不自带，可忽略此步

3. 安装 cuDNN（与 CUDA 11.8 兼容）

cuDNN 需要从 NVIDIA Developer 官网下载（需要账号登录）。你有两种常用安装方式：

方式 A：安装 .deb（推荐）

方式 B：安装 tar 包（更通用，适合自定义路径）

下载时请选择：cuDNN for CUDA 11.x（与 11.8 兼容）。
注意区分 runtime/dev/samples 包。

方式 A：使用 .deb 安装（推荐）

假设你下载到了以下文件（文件名可能随版本变化）：

cudnn-local-repo-ubuntu2204-8.x.x.x_1.0-1_amd64.deb

或者 NVIDIA 提供的分包：libcudnn8_*.deb、libcudnn8-dev_*.deb、libcudnn8-samples_*.deb

A1）本地 repo 包安装（如果你拿到的是 local-repo 包）
sudo dpkg -i cudnn-local-repo-ubuntu2204-*.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install libcudnn8 libcudnn8-dev

A2）分包安装（如果你拿到的是多个 .deb）

在包含这些 deb 的目录下执行：

sudo dpkg -i libcudnn8_*.deb
sudo dpkg -i libcudnn8-dev_*.deb
# 可选
sudo dpkg -i libcudnn8-samples_*.deb

方式 B：使用 tar 包安装（通用）

假设你下载的是：

cudnn-linux-x86_64-8.x.x.x_cuda11-archive.tar.xz

解压并拷贝到 CUDA 11.8：

tar -xf cudnn-linux-x86_64-*_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-*_cuda11-archive

sudo cp -P include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp -P lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
sudo ldconfig

4. 验证 cuDNN 是否安装成功
4.1 检查系统是否能找到 libcudnn
ldconfig -p | grep cudnn

4.2 检查 cuDNN 版本（以 header 为准）
grep -A 2 'CUDNN_MAJOR' /usr/local/cuda-11.8/include/cudnn_version.h

5. （可选）PyTorch / TensorFlow 简单验证
5.1 PyTorch（示例）

确保你安装的 PyTorch 对应 CUDA 11.8（例如 pip 安装时选择 cu118 轮子）。

验证：

python3 - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("runtime cuda:", torch.version.cuda)
PY

6. 常见问题排查
Q1：nvidia-smi 报错 / 显示 No devices were found

Secure Boot 未关闭导致驱动模块无法加载

驱动版本不匹配或安装残留冲突

内核头文件/ DKMS 未正确编译

建议：

sudo dmesg | grep -i nvidia | tail -n 50

Q2：nvcc 找不到

说明 PATH 没配好或 CUDA 没装在预期位置：

which nvcc
ls -l /usr/local/cuda-11.8/bin/nvcc

Q3：cuDNN 找不到（运行时报 libcudnn.so: cannot open shared object file）

确认：

echo $LD_LIBRARY_PATH
ls -l /usr/local/cuda-11.8/lib64/libcudnn*
sudo ldconfig
ldconfig -p | grep cudnn

Q4：驱动和 CUDA 版本要怎么搭？

驱动版本偏新一般没问题（新驱动通常兼容旧 CUDA runtime）

关键是：你安装的深度学习框架（PyTorch/TensorFlow）要匹配 CUDA/cuDNN 版本

7. 卸载（需要时）
7.1 卸载 CUDA Toolkit 11.8（APT 方式）
sudo apt -y remove --purge cuda-toolkit-11-8
sudo apt -y autoremove

7.2 卸载 NVIDIA 驱动
sudo apt -y purge 'nvidia*'
sudo apt -y autoremove
sudo reboot
