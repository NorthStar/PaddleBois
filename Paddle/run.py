import paddle

# 1
model = paddle.word2vec()

print("word 2 vec result:")
model.run('car','world')


print('------------------------------')
model2 = paddle.sentiment_classification()

print("sentiment classification")
model2.run([8,37,7])

# 2
#model3 = paddle.image_classification()
#res = model3.run('cat.png')



# 4
#model5 = paddle.machine_translation()

# 5
#model6 = paddle.recognize_digits()
#res = model6.run('img.png')

# 6
#model7 = paddle.object_detection()

