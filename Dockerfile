FROM tensorflow/tensorflow:latest
WORKDIR /home/MLPerf
RUN apt-get install -y python3 python3-pip
RUN pip3 install numpy scipy tensorflow
COPY . /home/MLPerf 
CMD [ "python3", "/home/MLPerf/test_runner.py" ]
