all: imagee push

image:
	sudo docker build -t aheirich/firsttest:latest .
	sudo docker tag aheirich/firsttest:latest aheirich/firsttest

push:
	sudo docker push aheirich/firsttest:latest
