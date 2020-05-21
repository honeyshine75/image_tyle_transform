FROM ubuntu:16.04

MAINTAINER honeyshine "me@imis.me"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev && \
    apt-get install -y nginx uwsgi uwsgi-plugin-python3
RUN pip3 install --upgrade pip  -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./requirements.txt /requirements.txt
COPY ./nginx.conf /etc/nginx/nginx.conf

WORKDIR /


RUN pip3 install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /

RUN adduser --disabled-password --gecos '' nginx\
  && chown -R nginx:nginx /app \
  && chmod 777 /run/ -R \
  && chmod 777 /root/ -R

ENTRYPOINT [ "/bin/bash", "/entry-point.sh"]