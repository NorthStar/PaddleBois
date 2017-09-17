import paddle

# 1
model = paddle.word2vec()
x = model.run('car','world')

model2 = paddle.sentiment_classification()
print(x)

# 2
model3 = paddle.image_classification()
#res = model3.run('cat.png')

# 3
model4 = paddle.sentiment_classification()
res = model4.run([8,37,7])


# 4
#model = paddle.machine_translation()

# 5
#model = paddle.recognize_digits()

# 6
#model = paddle.object_detection()

