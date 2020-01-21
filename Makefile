all:
	sudo docker build -t aheirich/firsttest:latest .
	sudo docker tag aheirich/firsttest:latest aheirich/firsttest
	sudo docker push aheirich/firsttest:latest
