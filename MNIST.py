import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron

def get_mnist():
    with np.load(f"train/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def convert_to_input(img):
    from PIL import Image
    im1 = Image.open(img)
    return ( 255 - np.array(im1.getdata(0)) )/ 255


images, labels = get_mnist()


## Define Perceptron

HiddenNodes = [784, 40, 10]
N = Perceptron(HiddenNodes)

training_inputs = images[:59500]
training_output = labels[:59500]
training_iterations = 4

test_inputs = images[59500:]
test_output = labels[59500:]

## Train

N.train(training_inputs, training_output, training_iterations)

## Test 1

img_from_file    = convert_to_input('test/1.png')
output_from_file =1

output = N.guess(img_from_file)
plt.imshow(img_from_file.reshape(28, 28), cmap="Greys")
plt.title(f"Is it a {output.argmax()}? It should be a {output_from_file}.");
plt.savefig('output/1_guessed.png')

# Test 2

index = np.random.randint(59500, 60000+1)

img = images[index]
lbl = labels[index]

output = N.guess(img)

plt.imshow(img.reshape(28, 28), cmap="Greys")
plt.title(f"Is it a {output.argmax()}? It should be a {lbl.argmax()}.")
plt.savefig('output/MNIST_guessed.png')