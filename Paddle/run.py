import paddle

# 1
model = paddle.word2vec()
x = model.run('car','world')

model2 = paddle.sentiment_classification()
print(x)

# 2
model = paddle.image_classification()

# 3
model = paddle.sentiment_classification()

# 4
model = paddle.machine_translation()

# 5
model = paddle.recognize_digits()

# 6
model = paddle.object_detection()

