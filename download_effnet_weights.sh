wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/randaug/efficientnet-b7-randaug.tar.gz

mkdir -p checkpoints/imagenet/effnet-b7
tar -xf efficientnet-b7-randaug.tar.gz -C checkpoints/imagenet/effnet-b7/
rm efficientnet-b7-randaug.tar.gz

mv checkpoints/imagenet/effnet-b7/efficientnet-b7-randaug/* checkpoints/imagenet/effnet-b7
