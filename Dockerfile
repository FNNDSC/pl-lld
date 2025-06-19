# Docker file for lld ChRIS plugin app
#
# Build with
#
#   docker build -t <name> .
#
# For example if building a local version, you could do:
#
#   docker build -t local/pl-lld .
#
# In the case of a proxy (located at 192.168.13.14:3128), do:
#
#    docker build --build-arg http_proxy=http://192.168.13.14:3128 --build-arg UID=$UID -t local/pl-lld .
#
# To run an interactive shell inside this container, do:
#
#   docker run -ti --entrypoint /bin/bash local/pl-lld
#
# To pass an env var HOST_IP to container, do:
#
#   docker run -ti -e HOST_IP=$(ip route | grep -v docker | awk '{if(NF==11) print $9}') --entrypoint /bin/bash local/pl-lld
#

FROM tensorflow/tensorflow:1.15.0-gpu-py3
#FROM gcr.io/deeplearning-platform-release/tf-cpu.1-15
LABEL maintainer="FNNDSC <dev@babyMRI.org>"

# download and unpack ML model weights
RUN curl -f https://stack.nerc.mghpcc.org:13808/swift/v1/AUTH_2dd3b02b267242d9b28f94a512ea9ede/fnndsc-public/weights/LLD/model.tar.gz \
     | tar --transform 's/^model/lld/' -xvz -C /usr/local/lib

# install dependencies and helpful (?) tools
COPY requirements.txt .
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC && \
     apt update && apt -y install pciutils sudo kmod libgl1-mesa-glx ffmpeg libsm6 libxext6
RUN pip install --upgrade pip setuptools wheel
RUN pip install pyOpenSSL --upgrade
RUN pip install -r requirements.txt


COPY . .
RUN pip install .

CMD ["lld", "--help"]
