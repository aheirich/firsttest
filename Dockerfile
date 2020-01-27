FROM tensorflow/tensorflow:latest
WORKDIR /home/MLPerf
RUN apt-get install -y python3 python3-pip
RUN pip3 install numpy scipy tensorflow
RUN chdir /home/MLPerf
COPY . .
RUN which python3
RUN pwd
RUN ls
CMD [ "python3", "./test_runner.py" ]
