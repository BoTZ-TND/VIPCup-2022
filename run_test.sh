#Script to execute the testing

## INPUT PARAMETERS
TEAM="team1" # name of team
CODE="submission.zip" # code of team in a zip file
TESTSET="$(pwd)/testset"
TESTSET_URL="https://www.grip.unina.it/download/vipcup2022/fewimage_for_codevalidation.zip" # folder of testset
IMAGE_URL="https://www.grip.unina.it/download/vipcup2022/Dockerfile_vipcup" # folder of testset
TESTSET_PASS="vippascup"

## DOWNLOAD DB and IMAGE
wget -nc --no-check-certificate -O testset.zip  "$TESTSET_URL" 
wget -nc --no-check-certificate -O Dockerfile_vipcup  "$IMAGE_URL" 
unzip -o -P "$TESTSET_PASS" testset.zip -d $TESTSET
nvidia-docker build -f Dockerfile_vipcup -t vipcup:0.1 .

## CREATE DIRECTORIES
ROOT_TEAM="$(pwd)/${TEAM}"
if [ -d "${ROOT_TEAM}/code" ]; then
    #Remove old files
    echo "Removing old files."
    NV_GPU=0 nvidia-docker run -v "${ROOT_TEAM}":/team gcr.io/kaggle-gpu-images/python:v115 rm -r /team/code
fi
mkdir -p "${ROOT_TEAM}/code"
mkdir -p "${ROOT_TEAM}/output"

## EXTRACT CODE
unzip -q $CODE -d "${ROOT_TEAM}/code/"

if [ ! -f "${ROOT_TEAM}/code/main.py" ]; then
    echo "ERROR: Python main does not exist."
    exit
fi

## EXECUDE CODE
echo RUNNING "${TEAM}"
NV_GPU=0 nvidia-docker run -v "${ROOT_TEAM}/code/":/code -v "${TESTSET}":/data:ro  -v "${ROOT_TEAM}/output":/output -w="/code" --network none vipcup:0.1 python main.py /data/metainfo.csv /output/out.csv
echo DONE
