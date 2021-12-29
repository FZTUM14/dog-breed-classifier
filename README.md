## Project "Dog breed classification"
### Description

The goal of this project is to build a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. 
So as a first step, the algorithm has to decide whether a human or dog is present in a given image. In a second step it then decides about the breed of the apparent dog or the most resembling dog breed for a human.

For solving the task of detecting human faces in images OpenCV's implementation of Haar feature-based cascade classifiers will be used. For identifying whether a dog is present in an image a Pre-trained VGG 16
net (trained on ImageNet) will be used. The performance of both models will be evaluated on the given datasets.

For classifying dog breeds, we will first try to train a small CNN from scratch. For that we will investigate necessary/useful pre-processing steps for training and test images and selection of appropriate optimizer, loss function and training hyperparameters. The trained model will be evaluated on the given test set. Of course with this approach (training from scratch) we can't expect a very good accuracy of the trained model since the chosen network architecture will be probably too simple to represent this complex task. Of course we could invest in more time and resources for training a larger model from scratch but this is not necessary: Luckily there are several pre-trained bigger models available that contain lots of useful information in earlier
layers that can be reused for the given problem. 
Therefore as a next step, transfer learning will be applied to get a better performing model for classifying dog breeds. Several pre-trained architectures will be finetuned on the given dataset of 133 dog breeds and evaluated on the test set. For performing transfer learning in a lightweight manner I will use Pytorch Lightning and will finetune a pre-trained Mobilenet, Resnet50 and
ResNext101. Of course this again includes appropriate preprocessing of the images, selection of appropriate loss functions and optimizers and training hyperparameters.

Finally we'll put together the algorithms for identifying human faces, dogs and classifying dog breeds to build the final application and test it on some images the algorithm has never seen during training.


### Instructions

1. Clone the repository and navigate to the downloaded folder.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the requirements.txt in the repository.
5. Open a terminal window and navigate to the project folder. Open the notebook dog_app.ipynb and follow the instructions.
