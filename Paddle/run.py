import paddle
 
model = paddle.word2vec()
x = model.run('car','world')

model2 = paddle.sentiment_classification()
print(x)
