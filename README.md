# Chord Buddy

## Inspiration ##
Learning and memorizing chords for the guitar is integral for beginner guitar players. Chords are the building blocks of most songs and so learning them makes playing songs much easier. We wanted to create a tool that would make it easier to guitar players to memorize chords and to be told when their finger placement or sound is incorrect. 

## What It Does ##
The application displays chords on the screen for the user to play and then uses two neural networks to determine if the shape and sound of the chord the user played is correct. 

The application has a running video stream that the user can see themselves through. It will flash a certain chord on the screen, after which the user is given a few seconds to get ready and then must play that chord. If they got it correct, their score will increase and the application will display that they were correct before moving on to the next chord.  

## How It Works ##
A picture and sound recording is taken of the user playing the chord. The picture is run through a VGG neural network model with four convolutional layers and two dense layers. We added a flatten and dropout layer to prevent overfitting since we have limited data and we used SoftMax activation and categorical cross entropy loss.  We trained the model on our data using an augmentation configuration to prevent overfitting. 

## How To Use It ##
Download the source code and then run python userPart.py in terminal to run the application. 

## Built With ##
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/)
