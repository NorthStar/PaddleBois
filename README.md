# PaddleBois


## CAll Server 
ssh -i hackmit-paddlepaddle-1.pem ubuntu@35.167.14.53

# starting docker server
(see init())
#Ending docker server (example)
docker rm -f e50ad077c9e7006b3ffef2f0e888c0700623c9ca69ca432227fe590e32039424

## Structure

### init()
1. Download files
2. scp to server
3. go into or create model/mode-name folder
4. start docker or nvidia docker server (see PaddlePaddle documentation for each mode) 
nvidia-docker run --name {{SERVER NAME}} -d -v $PWD:/data -p {{PORT NUM}}:80 -e WITH_GPU=1 paddlepaddle/book:serve-gpu

### run()
1 create request according to PaddlePaddle format
2. send request to server
3. return response

### destruct()
Quit the docker servers
 
