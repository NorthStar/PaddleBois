# EZPaddle

EZPaddle is a framework to allow you to use Paddle's pre-trained models with jsut one line of code.

```
model = paddle.sentiment_classification()
model.run([8,37,7])
```
## Parameters
Set the parameters in paddle.py
```
hostname = 35.167.14.53
sentiment_analysis_port = 6000
machine_translation_port = 2000
image_recognition_port = 3000
```


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
 
