# CNN for behavior tracking


Using a convolutional neural network to score rodent behavior v0.2

##The problem:
One of the more fundamental behavioral tasks in modern day neuroscience is a class of tasks broadly classified as recognition tasks. In these tasks a researcher times how long an animal interacts with a number of objects.  As most animals find novelty rewarding animals will interact with a moved object (object place memory, hippocampal), a novel object (object recognition memory, probably entorhinal), a novel mouse (social recognition memory, region currently being determined but is disrupted in models of Fragile X), or a temporally less recent object (CA3 dependent) more than an unchanged one. 
While this sounds fairly trivial from a computer vision perspective:

1.	Assign a circle around an object
2.	background subtract 
3.	profit

It turns out that animals can interact with objects in non-attentive ways (climbing on objects, chewing on objects, grooming near objects) and can interact with objects from various distances. Trained researchers have little or no ability distinguishing this behavior but new researchers often struggle. Also, given a single experiment can involve 4 training sessions (5min each) and a test session (3-5mins) and 40+ rodents, the time required for trained personnel is immense. 

If a machine learning algorithm could be developed with even reasonable accuracy it could be used to at least automate scoring of the training sessions (which are usually used to determine ‘did a mouse interact with both objects an approximately similar amount). If an algorithm could be developed that can cross correlate with a skilled researcher 95%+ it would mat about what we see with humans.

##The Dataset:

Normally to train a proper CNN thousands of examples of labeled data, or a pretrained network on Imagenet would be required. But I’m worried that imagenet trained CNNs seem to ditch spatial information (that’s what they are trained to do!), which is actually massively important for this task. 

Luckily videos generate a bunch of free training data; one 5 minute video generates 4500 unique images. We can align this with a person scoring a video by building a stop watch program that captures this info. A single video then could give us enough data to prototype this on (not well, but it won’t be bad)

##Approach: 

The videos from this set came in sets of 3

![png][sample_image.png]

(Note animals in this experiment were all handled by UCI IACUC protocols and animals under cups had no signs of distress and were only under the cups for minimal amounts of time, there was also a fair amount of space there)

Video slicer slices the videos into 3 equally sized sub videos based on human found borders (next version find these automatically). It then uses .trn files which say if an animal was interacting with an object. It splits these videos into 3 categories: top object, bottom, object, and no object.

##Gotchas!

Of course it wasn’t that easy, humans are notably ‘slow’ relative to a computer (300-500ms!) to fix this, I found that empirically you need to throw a 7 frame buffer on the videos, I also found performance was enhanced by dropping these frames as variable reaction speed/catching up that humans do lead to that being the safe choice.

##FINALLY THE MODEL:

4 convolutional layers, then a max pooling layer and a dense layer, boring, kind of deep but  for the size of the dataset and the ‘simplicity’ of the task and the relatively few number of tasks, and as we train it we see:

![png][train_acc_base.png]
![png][train_loss_base.png]

Pretty standard training, approaching 90%!
Using VGG19 to see if a pertained network works better:
It certainly hits criterion faster but has some overfitting (not surprising I only have 1300 images per category!)
 
![png][train_acc_VGG19.png]
![png][train_loss_VGG19.png]

Graphs are great but it does lead to the biggest question: how does it look?

![](mo1.gif)

Note, this is on the data it was trained as so it's not the most valid. Red is a top interaction, blue is no interaction, green is botom interaction.

What do we need to do to take this into production?
1.	Way more training data (We have that!)
2.	A way to automatically find the walls. Pretty simple segmentation task as they will positively glow in an edge detector.
3.	Multiple mouse type and behaviors? Is the algorithm universal or do we need to ‘retrain it’ per behavior/mouse type. 
4.	Can it replicate effects. Replicating approximate human scoring is great for a lot of things, but if it can’t replicate group differences it might be missing essential and subtle components of the behavior
5.	Can we beat 95%
6.	Hyper parameter fitting (maybe not hugely important given that vgg19 and a stupid simple network did about the same but see problem 1)
