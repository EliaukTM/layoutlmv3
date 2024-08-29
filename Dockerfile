FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

LABEL Author="haymai<github.com/haymaicc>"


COPY . /work
WORKDIR /work

RUN apt-get update && apt-get install -yq --no-install-recommends python3 python3-dev python3-pip

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000

EXPOSE 8099

CMD python3 run.py