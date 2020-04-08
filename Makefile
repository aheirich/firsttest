all: image push nvidia-image nvidia-push

image: Dockerfile
	sudo docker build -t aheirich/firsttest:latest .
	sudo docker tag aheirich/firsttest:latest aheirich/firsttest

nvidia-image: Dockerfile.nvidia
	sudo docker build -f Dockerfile.nvidia -t aheirich/firsttest-nvidia:latest .
	sudo docker tag aheirich/firsttest-nvidia:latest aheirich/firsttest-nvidia

tpu-image: Dockerfile.tpu
	sudo docker build -f Dockerfile.tpu -t aheirich/firsttest-tpu:latest .
	sudo docker tag aheirich/firsttest-tpu:latest aheirich/firsttest-tpu


BUCKET_NAME="myBucket"
REGION=us-west1
ZONE=${REGION}-a

bucket:
	gsutil mb -l ${REGION} gs://${BUCKET_NAME}

#PROJECT_ID=$(gcloud config list project --format "value(core.project)")
PROJECT_ID=firsttest-268800
IMAGE_REPO_NAME=mlperfsinglechip_image
IMAGE_TAG=latest
export IMAGE_URI_BASE=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}
export IMAGE_URI=${IMAGE_URI_BASE}:${IMAGE_TAG}

gcloud-image: Dockerfile.gcloud
	sudo docker build -f Dockerfile.gcloud -t aheirich/firsttest-gcloud:latest .
	sudo docker tag aheirich/firsttest-gcloud:latest aheirich/firsttest-gcloud
#	sudo docker build -f Dockerfile.gcloud -t ${IMAGE_URI} .
#	sudo docker tag ${IMAGE_URI_BASE}:latest ${IMAGE_URI_BASE}


push:
	sudo docker push aheirich/firsttest:latest

nvidia-push:
	sudo docker push aheirich/firsttest-nvidia:latest

nvidia-pull:
	singularity pull docker://aheirich/firsttest-nvidia:latest

gcloud-push:
	sudo docker push aheirich/firsttest-gcloud:latest
#	sudo docker push ${IMAGE_URI}

JOB_NAME=${IMAGE_REPO_NAME}_job
MODEL_DIR=modeldir

gcloud-submit:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--region ${REGION} \
		--master-image-uri ${IMAGE_URI} \
		-- \
		--model-dir=gs://${BUCKET_NAME}/${MODEL_DIR} 

create-compute-instance:
	gcloud compute instances create-with-container fubar-nv-v100 --accelerator type=nvidia-tesla-v100,count=1 --zone ${ZONE} --container-image aheirich/firsttest-gcloud:latest --maintenance-policy TERMINATE

gpu-list:
	gcloud compute accelerator-types list

ssh-compute-instance:
	gcloud beta compute ssh --zone "northamerica-northeast1-c" "fubar1" --project "firsttest-268800"
#	gcloud compute ssh fubar1 --container aheirich/firsttest-gcloud:latest

