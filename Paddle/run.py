import paddle
 
model = paddle.word2vec()
x = model.run('car','world')
print(x)
