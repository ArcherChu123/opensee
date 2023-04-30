
# 创建基于CNN的机器视觉识别模型
# 1. 导入必要的包
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# 2. 导入并准备数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# 将像素值缩放到0-1之间
train_images, test_images = train_images / 255.0, test_images / 255.0
# 3. 验证数据
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck']
# 4. 查看训练集中的前25张图片，并显示类别名称
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
# 5. 构建卷积神经网络
model = models.Sequential()
# 第一层卷积层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 第一层池化层
model.add(layers.MaxPooling2D((2, 2)))
# 第二层卷积层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 第二层池化层
model.add(layers.MaxPooling2D((2, 2)))
# 第三层卷积层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 第三层池化层
model.add(layers.MaxPooling2D((2, 2)))
# 6. 在模型顶部添加全连接层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# 8. 查看模型的架构
model.summary()
# 9. 编译并训练模型
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# 10. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
# 11. 绘制准确率和损失值曲线
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# 12. 绘制损失值曲线
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# 13. 使用模型进行预测
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
# 14. 验证预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()
# 15. 保存模型
model.save('cifar10_model.h5')
# 16. 加载模型
new_model = tf.keras.models.load_model('cifar10_model.h5')
new_model.summary()
# 17. 验证加载的模型
probability_model = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
# 18. 验证预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()
# 19. 使用模型进行预测
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
# 20. 验证预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()

