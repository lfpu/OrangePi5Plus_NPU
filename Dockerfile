FROM ubuntu:20.04
 
# 安装基础依赖
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    libopencv-dev python3-opencv \
    git wget unzip && \
    pip3 install --upgrade pip
 
# 安装 rknn-toolkit2
RUN pip3 install rknn-toolkit2
 
# 设置工作目录
WORKDIR /workspace
 
CMD ["/bin/bash"]