FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
MAINTAINER Yi Dong <yidong80@gmail.com>

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev

#
# Python 3.5
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib Cython requests

#
# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888


#
# Tensorflow - GPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow-gpu

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4.1
#
RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall
# Get source from github
RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

#
# Caffe
#
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget libatlas-base-dev \
        libboost-all-dev libgflags-dev libgoogle-glog-dev \
        libhdf5-serial-dev libleveldb-dev liblmdb-dev \
        libopencv-dev libprotobuf-dev libsnappy-dev protobuf-compiler \
        python-dev python-numpy python-pip python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
ARG CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig





# install MaskRCNN
# Get source from github
RUN git clone https://github.com/matterport/Mask_RCNN.git /usr/local/src/maskrcnn
# Dependencies
RUN pip3 --no-cache-dir install -r /usr/local/src/maskrcnn/requirements.txt
# Install
RUN cd /usr/local/src/maskrcnn/ && \
    python3 /usr/local/src/maskrcnn/setup.py install && \
    cd /
# Enivronment variables
ENV PYTHONPATH=/usr/local/src/maskrcnn/mrcnn:$PYTHONPATH


WORKDIR "/root"
CMD ["/bin/bash"]