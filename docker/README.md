# Docker

1. create docker image
   `$ docker build ./docker -t <your_image_name>`
   example: `$ docker build ./docker -t dance_env`
1. run created docker image `docker run -gpus all <your_image_name>`
   example: `$ docker run -gpus all dance_env`
