all: image push nvidia-image nvidia-push

image:
	sudo docker build -t aheirich/firsttest:latest .
	sudo docker tag aheirich/firsttest:latest aheirich/firsttest

push:
	sudo docker push aheirich/firsttest:latest

nvidia-image:
	sudo docker build -f Dockerfile.nvidia -t aheirich/firsttest-nvidia:latest .
	sudo docker tag aheirich/firsttest-nvidia:latest aheirich/firsttest-nvidia

nvidia-push:
	sudo docker push aheirich/firsttest-nvidia:latest

nvidia-pull:
	singularity pull docker://aheirich/firsttest-nvidia:latest
