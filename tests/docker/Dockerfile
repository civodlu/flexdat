FROM alpine:3.14

#RUN mkdir -p /inputs /outputs
#RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
#RUN chown -R algorithm:algorithm /opt/algorithm /inputs /outputs
#USER algorithm

ENTRYPOINT cp -r /inputs/* /outputs

#   docker build -t flexdat_test -f Dockerfile .
#   docker rmi flexdat_test:latest -f 
#   docker run -it -v "/home/ludovic/tmp/docker/inputs:/inputs" -v "/home/ludovic/tmp/docker/outputs:/outputs" --entrypoint /bin/sh  flexdat_test:latest
#   docker run -v "/home/ludovic/tmp/docker/inputs:/inputs" -v "/home/ludovic/tmp/docker/outputs:/outputs" flexdat_test:latest
