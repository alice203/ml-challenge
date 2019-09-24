# Machine Learning Challenge

The task was to learn to recognize whether an image of a handwritten digit and a recording of a spoken digit refer to the same or different number. „False“ defined the case where the image and the recording refered to different numbers and „true“ defined the case where the image and the recording referred to the same number. 
Each image was given as 784-dimensional vector, which represented a 28x28 pixel grayscale image. Pixel intensities ranged from 0 (black) to 255 (white). Each sound recording of a spoken name of a digit (e.g. “zero”, “one” etc, pronuounced in Arabic) was given as an array of pre-extracted audio features, so called Mel-Frequency Cepstral Coefficients (MFCC). These features encode the characteristics of a 10 milisecond frame of speech. Each recording is of variable length, and thus each example was given as an array of shape (N, 13), where N is the number of frames in the recording, and 13 the number of MFCC features. 
The dataset was made available in numpy array format:

•	written_train.npy: array with 45,000 rows and 784 columns

•	written_test.npy: array with 15,000 rows and 784 columns

•	spoken_train.npy: array with 45,000 rows. Each row is an object of shape (N, 13)

•	spoken_test.npy: array with 15,000 rows. Each row is an object of shape (N, 13)

•	match_train.npy: array with 45,000 boolean values (False or True)


The evaluation metric for this task was error rate accuracy and the submission file had to be an array of 15,000 boolean values, specifying whether the images and sounds from the test data are matched or not. 
