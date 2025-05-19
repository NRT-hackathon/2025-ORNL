# %% [markdown]
# # Deep Learning Analysis of Defect and Phase Evolution During Electron Beam Induced Transformations in WS$_2$

# %% [markdown]
# ### Authors: Artem Maksov, Maxim Ziatdinov*

# %% [markdown]
# *_Correspondence to: ziatdinovmax@gmail.com_
# 
# _Date: 03/09/2018_     

# %% [markdown]
# In this notebook, we demonstrate a complete workflow for extraction and classification of defects in a STEM image stack (STEM "movie"). 

# %% [markdown]
# ## Table of Contents 

# %% [markdown]
# 1. We use a single frame from a movie with FFT subtraction method to create a training dataset. 
# 2. We train a fully convolutional neural network (without "dense" layers) to find defects that break lattice periodicity.
# 3. We apply the network to the whole movie and extract subwindows containing individual defects.
# 4. We train a Gaussian Mixture Model to classify the defects and then we explore their structure and evolution in time.

# %% [markdown]
# Requirenments: 
#     
# 1. Python 3 (matplolib, numpy, scipy, sklearn, opencv, h5py)
# 2. Latest version of pydm3reader: https://microscopies.med.univ-tours.fr/?page_id=260 
# 3. Keras + tensorflow backend

# %% [markdown]
# ## 0. Imports 

# %%
#I/O
import os
import dm3_lib as dm3
import h5py
import joblib

#graphics
import matplotlib
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
import pylab as P


#processing 
import numpy as np
from scipy import fftpack
from scipy import ndimage

from skimage.feature import blob_log
from sklearn import cluster, mixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from collections import OrderedDict
from scipy.spatial import cKDTree

# %%
#%matplotlib nbagg
#%matplotlib inline

# %%
os.environ["KERAS_BACKEND"] = "torch"

# %%
import cv2
#neural networks
from keras import layers
from keras import models
from keras import callbacks
# from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
# from keras.layers import Dropout, Activation, Reshape
# from keras.models import Model, load_model
# from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# %% [markdown]
# ## 1. Initial image processing 

# %% [markdown]
# ### 1.1. Reading in the file 

# %%
def extract_frames(video_path):
    """
    Extract frames from a grayscale video file into a list of numpy arrays.
    
    Parameters:
    video_path (str): Path to the .avi video file
    
    Returns:
    list: List of numpy arrays, where each array represents a grayscale frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"Frame count: {frame_count}")
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    frames = []
    
    # Read frame by frame
    while True:
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
            
        # Convert frame to grayscale if it's not already
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    return frames

# %% [markdown]
# Define path to the file: 

# %%
path = './ORNL-DeepLearningForAtomicScaleDefectTracking/supplementary/'
#filename = 'stack4.dm3'
filename = '41524_2019_152_MOESM2_ESM.avi'

# %% [markdown]
# Read in the image data:

# %%
#dm3f = dm3.DM3(path + filename)
#imgdata = dm3f.imagedata
imgdata = extract_frames(path + filename)

# %% [markdown]
# We need to rescale the image according to gain, and normalize it:

# %%
# for t in dm3f.tags:
#     if 'PMTDF' in t:
#         scale = float(dm3f.tags[t])/64000
        
# imgdata = imgdata/scale

# %%
# imgdata = imgdata - np.amin(imgdata)
# imgdata = imgdata/np.amax(imgdata)

# %% [markdown]
# Create directory for storing outputs:

# %%
directory = './Figures/' + filename[:-4] + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

# %% [markdown]
# ### 1.2 FFT subtraction 

# %% [markdown]
# We will explore the first frame of the movie to estimate parameters for FFT subtraction

# %%
imgsrc = imgdata[0]

# %% [markdown]
# #### 1.2.1 Plotting real space image:

# %%
#plot real space image
fig100 = plt.figure(100, figsize=(5,5))
ax = fig100.add_axes([0, 0, 1, 1])

ax.imshow(imgsrc, cmap = 'gray')
ax.axis('off')

#save the figure
fig100.savefig(directory + filename + '_realspace.png')

# %% [markdown]
# #### 1.2.2 FFT subtraction

# %% [markdown]
# We perform FFT subtraction in a similar manner as to described in (reference)

# %% [markdown]
# We need to create a mask and apply it as a high-pass filter in the frequency domain. Currently, this depends on 1 human-defined parameter: 1/ratio of radius of the mask to the image size.

# %%
# 1/ratio of the radius of the mask to the original image in pixels
maskratio = 10

# %%
def FFTmask(imgsrc, maskratio=10):
    """Takes a square real space image and filter out a disk with radius equal to:
    1/maskratio * image size.
    Retruns FFT transform of the image and the filtered FFT transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2((imgsrc)) 
    # Now shift so that low spatial frequencies are in the center.
    F2 = (fftpack.fftshift((F1)))
    # copy the array and zero out the center
    F3 = F2.copy()
    l = int(imgsrc.shape[0]/maskratio)
    m = int(imgsrc.shape[0]/2)
    y,x = np.ogrid[1: 2*l + 1, 1:2*l + 1]
    mask = (x - l)*(x - l) + (y - l)*(y - l) <= l*l
    F3[m-l:m+l, m-l:m+l] = F3[m-l:m+l, m-l:m+l] * (1 - mask)
    
    return F2, F3

# %%
F2, F3 = FFTmask(imgsrc, maskratio)

# %% [markdown]
# We can plot both FFT and filtered FFT to see how it performs:

# %%
#plot 
fig101 = plt.figure(101)
ax1 = fig101.add_subplot(121)
ax1.imshow(np.abs(np.log(F2)), cmap='hot')
ax1.axis('off')
ax2 = fig101.add_subplot(122)
masked_array = np.ma.array (np.abs(np.log(F3)), mask=np.isnan(np.abs(np.log(F3))))
cm = matplotlib.cm.hot
cm.set_bad('black',1.)
ax2.imshow(masked_array, cm)
ax2.axis('off')

fig101.savefig(directory + 'FFTfilter.png')

# %% [markdown]
# Now we reconstruct the real space image from filtered FFT image and subtract it from the original to identify locations with broken symmetry 

# %%
def FFTsub(imgsrc, F3):
    """Takes real space image and filtred FFT.
    Reconstructs real space image and subtracts it from the original.
    Returns normalized image. 
    """
    reconstruction = np.real(fftpack.ifft2(fftpack.ifftshift(F3)))
    diff = np.abs(imgsrc - reconstruction)
    
    #normalization
    diff = diff - np.amin(diff)
    diff = diff/np.amax(diff)
    
    return diff

# %%
diff = FFTsub(imgsrc, F3)

# %%
fig102 = plt.figure(102, figsize=(5,5))
ax = fig102.add_axes([0, 0, 1, 1])
ax.imshow(diff, cmap='gray')
ax.axis('off')
fig102.savefig(directory + 'diff.png')

# %% [markdown]
# #### 1.2.3 Creating a defect location map

# %% [markdown]
# We need to thresold the resulting difference image to create a map for the defect locations, which will serve as labels for pixels in our training set

# %% [markdown]
# We need to define 2 thresholds: low for low intensity defects and high for high-intensity. Since we normalized the difference image they are both in [0;1) range

# %% [markdown]
# We can use histogram of intensities to inform our choice:

# %%
diffall = diff.reshape(diff.shape[0] * diff.shape[1])

# %%
fig103 = plt.figure(103)
ax103 = fig103.add_axes([0, 0, 1, 1])
ax103.hist(diffall, bins='auto');

# %% [markdown]
# Or we can inform our choice by computing statistics for the intensity distributions, but those should be taken as guidance and confirmed by visual inspection as they can vary from image to image

# %%
print('Defining cutoff as 2 std, low threshold:', np.mean(diffall) - 2* np.std(diffall), 
      ' high thresold: ', np.mean(diffall) + 2* np.std(diffall))
print('Defining cutoff as 3 std, low threshold:', np.mean(diffall) - 3* np.std(diffall), 
      ' high thresold: ', np.mean(diffall) + 3* np.std(diffall))

# %% [markdown]
# Define the thresholds:

# %%
def threshImg(diff, threshL=0.25, threshH=0.75):
    """Takes in difference image, low and high thresold values, and outputs a map of all defects.
    """
    
    threshIL = diff < threshL  
    threshIH = diff > threshH
    threshI = threshIL + threshIH
    
    return threshI

# %%
threshL = 0.25
threshH = 0.75

# %% [markdown]
# Plot the results:

# %%
fig104 = plt.figure(104, figsize=(5,5))
ax = fig104.add_axes([0, 0, 1, 1])

threshI = threshImg(diff, threshL, threshH)

ax.imshow(imgsrc)
ax.imshow(threshI, cmap='gray', alpha=0.3)
ax.axis('off')

# %%
labelIm = threshI.astype(int)

# %% [markdown]
# #### 1.2.4. Generating training set 

# %% [markdown]
# We can generate training set by taking patches of the image and corresponding labels

# %% [markdown]
# We define helper functions:

# %%
#Generate x, y positions for sliding windows
def GenerateXYPos(window_size, window_step, image_width):
    """Takes the window size, step, and total image width 
    and generates all xy pairs for sliding window"""   
    xpos_vec = np.arange(0,image_width-window_size,window_step)
    ypos_vec = np.arange(0,image_width-window_size,window_step)
        
    num_steps = len(xpos_vec)    
            
    xpos_mat = np.tile(xpos_vec, num_steps)
    ypos_mat = np.repeat(ypos_vec, num_steps)
    pos_mat = np.column_stack((xpos_mat, ypos_mat))
            
    return pos_mat

# %%
def MakeWindow(imgsrc, xpos, ypos, window_size):
    """returns window of given size taken at position on image"""
    imgsrc = imgsrc[xpos:xpos+window_size, ypos:ypos+window_size]
    return imgsrc

# %%
def imgen(raw_image, pos_mat, window_size):
    """Returns all windows from image for given positions array"""
    
    immat = np.zeros(shape = (len(pos_mat), window_size, window_size))
    
    for i in np.arange(0,len(pos_mat)):
        img_window = MakeWindow(raw_image, pos_mat[i,0], pos_mat[i,1], window_size) 
        immat[i,:,:,] = img_window
    
    return immat

# %%
def image_preprocessing(image_data, norm=0):
    """Reshapes data and optionally normalizes it"""
    image_data = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], 1)
    image_data = image_data.astype('float32')
    if norm != 0:
        image_data = (image_data - np.amin(image_data))/(np.amax(image_data) - np.amin(image_data))
    return image_data

# %%
def label_preprocessing(image_data, nb_classes):
    """Returns labels / ground truth for images"""
    
    label4D = np.empty((0, image_data.shape[1], image_data.shape[2], nb_classes))
    for idx in range(image_data.shape[0]):
        img = image_data[idx,:,:]
        n, m = img.shape
        enc = OneHotEncoder()
        img = np.array(enc.fit_transform(img.reshape(-1,1)).todense())
        img = img.reshape(n, m, nb_classes)
        label4D = np.append(label4D, [img], axis = 0)    
    return label4D

# %%
def label_preprocessing2(image_data):
    """we can simplify this in case of only two classes"""
    
    label1 = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], 1)
    label2 = -label1 + 1
    label4D = np.concatenate((label2, label1), axis = 3)
    return label4D

# %% [markdown]
# We can control the number of patches by changing window size and window step

# %%
window_size = 256
window_step = 8
pos_mat = GenerateXYPos(window_size, window_step, imgsrc.shape[0])
print("On image size: ", imgsrc.shape[0], ' by ',  imgsrc.shape[1], ' pixels: ', len(pos_mat), "patches will be generated")

# %%
immat = imgen(imgsrc, pos_mat, window_size)
labelmat = imgen(labelIm, pos_mat, window_size)

# %%
immat2 =  image_preprocessing(immat, 1)
labelmat2 =  label_preprocessing(labelmat, nb_classes = 2)

# %% [markdown]
# ## 2. Training the Neural Network model

# %% [markdown]
# First, we need to define network architecture:

# %%
def model_defect(input_img, nb_classes = 2):
    """Creates a Deep Learning model for defect identification"""
    
    x = layers.Convolution2D(20, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Convolution2D(40, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
        
    x = layers.Convolution2D(40, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Convolution2D(20, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Convolution2D(nb_classes, (3, 3), activation = 'linear', padding='same')(x)
    x = layers.Convolution2D(nb_classes, (1, 1), activation = 'linear', padding='same')(x)

    output = layers.Activation('softmax')(x)
    
    return models.Model(input_img, output)

# %% [markdown]
# We can split data into training and validation set to improve training:

# %%
nb_classes = 2
target_size = (256, 256)

X_train, X_test, y_train, y_test = train_test_split(immat2, labelmat2, test_size = 0.2, random_state = 42)

# %% [markdown]
# Finally, we train the model:

# %%
input_img = layers.Input(shape=(target_size[0], target_size[1], 1)) 

model = model_defect(input_img)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
callback_tb = callbacks.TensorBoard(log_dir='/tmp/AtomGen', histogram_freq=0,
                          write_graph=True, write_images=False)

model.fit(X_train, y_train, epochs=50, batch_size=32, 
          validation_data=(X_test, y_test), shuffle=True, verbose = 1)

# %% [markdown]
# And save it:

# %%
model.save("./DefectModel3.h5")
model.save_weights("./DefectModel3.weights.h5")
print('Saved model and weights to disk.\n')

# %% [markdown]
# ## 3. Processing the movie

# %% [markdown]
# We can apply the trained model to the rest of the movie to identify defects.

# %% [markdown]
# ### 3.1 Processing single frame

# %%
nframes = len(imgdata)

# %%
def splitImage(img, target_size):
    """Splits image into patches with given size"""
    
    xs = img.shape[0]
    ys = img.shape[1]
    nxp = int(xs/target_size[0])
    nyp = int(xs/target_size[1])
    
    impatchmat = np.zeros(shape=(int(nxp*nyp), target_size[0], target_size[1], 1))
    
    count = 0
    for i in range(nxp):
        for i2 in range(nyp):
            xstart = target_size[0] * i
            ystart = target_size[1] * i2
            xend = target_size[0] * (i + 1)
            yend = target_size[1] * (i2 + 1)

            impatchmat[count, :, :, 0] = img[xstart:xend, ystart:yend]
            count = count + 1
            
    return impatchmat

# %%
def predictDefects(img, model, target_size, nb_classes=2):
    """Uses given DL model to generate prediciton maps on the image"""
    
    xs = img.shape[0]
    ys = img.shape[1]
    nxp = int(xs/target_size[0])
    nyp = int(xs/target_size[1])
    classpred = np.zeros(shape=(nb_classes,xs, ys))

    impatchmat = splitImage(img, target_size)
    res = model.predict(impatchmat)
    
    count = 0
    for i in range(nxp):
        for i2 in range(nyp):
            xstart = target_size[0] * i
            ystart = target_size[1] * i2
            xend = target_size[0] * (i + 1)
            yend = target_size[1] * (i2 + 1)
            
            for i3 in range(nb_classes):
                classpred[i3, xstart:xend, ystart:yend] = res[count, :, :, i3]
                
            count = count + 1
            
    return classpred

# %%
def predictDefectsOverlap(img, model, target_size, nb_classes=2):
    """Uses given DL model to generate prediction maps on the image with overlapping windows.
    
    Parameters:
    img: Input image array
    model: Trained keras model
    target_size: Tuple of (height, width) for the window size
    nb_classes: Number of classes in prediction
    """
    xs = img.shape[0]
    ys = img.shape[1]
    window_size = target_size[0]  # Assuming square windows
    window_step = window_size // 2  # Use 50% overlap
    
    # Initialize accumulator arrays
    predictions = np.zeros((nb_classes, xs, ys))
    counts = np.zeros((xs, ys))
    
    # Generate window positions
    xpos_vec = np.arange(0, xs - window_size + 1, window_step)
    ypos_vec = np.arange(0, ys - window_size + 1, window_step)
    
    # Create all possible combinations
    xv, yv = np.meshgrid(xpos_vec, ypos_vec)
    pos_mat = np.stack((xv.flatten(), yv.flatten()), axis=1)
    
    # Process each window
    for pos in pos_mat:
        x_start, y_start = int(pos[0]), int(pos[1])
        x_end = x_start + window_size
        y_end = y_start + window_size
        
        # Extract and process window
        window = img[x_start:x_end, y_start:y_end]
        window = window.reshape(1, window_size, window_size, 1)
        window = window.astype('float32')
        window = (window - np.amin(window))/(np.amax(window) - np.amin(window))
        
        # Get prediction for window
        pred = model.predict(window, verbose=0)
        
        # Accumulate predictions and counts
        for i in range(nb_classes):
            predictions[i, x_start:x_end, y_start:y_end] += pred[0, :, :, i]
        counts[x_start:x_end, y_start:y_end] += 1
    
    # Average predictions by count of overlapping windows
    for i in range(nb_classes):
        predictions[i] /= np.maximum(counts, 1)  # Avoid division by zero
        
    return predictions

# %% [markdown]
# Here is just a test for a single frame:

# %%
fnum = 80
classpred = predictDefectsOverlap(imgdata[fnum], model, target_size, nb_classes)

# %%
fig300 = plt.figure(300)
ax1 = fig300.add_subplot(121)
ax1.imshow(imgdata[fnum])
ax1.axis('off')
ax2 = fig300.add_subplot(122)
ax2.imshow(classpred[1])
ax2.axis('off')

# %%
fig301 = plt.figure(301, figsize=(5,5))
ax = fig301.add_axes([0, 0, 1, 1])

ax.imshow(imgdata[fnum])
ax.imshow(classpred[1] > 0.99, cmap='hot', alpha=0.5)
ax.axis('off')

# %% [markdown]
# ### 3. 2 Defect location identification

# %% [markdown]
# To classify the defects, we need to extract image patches containing them. This means we need to filter out defects with large areas, which can be observed in the second half of the movie.

# %%
softmax_threhold = 0.8

# %%
def extractDefects(img, classpred, softmax_threhold = 0.5, bbox=16):
    
    defim = np.ndarray(shape=(0, bbox*2, bbox*2))
    defcoord = np.ndarray(shape=(0, 2))
    defcount = 0
    
    _, thresh = cv2.threshold(classpred[1], softmax_threhold, 1, cv2.THRESH_BINARY)
    
    s = [[1,1,1],[1,1,1],[1,1,1]]
    labeled, nr_objects = ndimage.label(thresh, structure=s) 
    loc = ndimage.find_objects(labeled)
    cc = ndimage.measurements.center_of_mass(labeled, labeled, range(nr_objects + 1))
    sizes = ndimage.sum(thresh,labeled,range(1, nr_objects +1)) 

    #filter found points
    cc2 = cc[1:]
    t = zip(cc2, sizes)
    ccc = [k[0] for k in t if (k[1] < 700 and k[1] > 5) ]

    max_coord = ccc
    
    
    for point in max_coord:
    
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round( point[1] - bbox))
        endy = starty + bbox*2

        if startx > 0 and startx < img.shape[0] - bbox*2:
            if starty > 0 and starty < img.shape[1] - bbox*2: 

                defim.resize(defim.shape[0] + 1, bbox*2, bbox*2)
                defim[defcount] = img[startx:endx, starty:endy]
                defcoord.resize(defcoord.shape[0] + 1, 2)
                defcoord[defcount] = point[0:2]

                defcount = defcount + 1
            
    
    return thresh, defim, defcoord

# %%
def extractDefectsOverlap(img, predictions, softmax_threshold=0.5, bbox=16, min_area=5, max_area=700):
    """Extract defects with additional filtering for connected components"""
    
    defim = np.ndarray(shape=(0, bbox*2, bbox*2))
    defcoord = np.ndarray(shape=(0, 2))
    defcount = 0
    
    # Threshold predictions
    _, thresh = cv2.threshold(predictions[1], softmax_threshold, 1, cv2.THRESH_BINARY)
    
    # Find connected components
    s = [[1,1,1],[1,1,1],[1,1,1]]
    labeled, nr_objects = ndimage.label(thresh, structure=s)
    loc = ndimage.find_objects(labeled)
    cc = ndimage.measurements.center_of_mass(labeled, labeled, range(nr_objects + 1))
    sizes = ndimage.sum(thresh, labeled, range(1, nr_objects + 1))
    
    # Filter found points
    cc2 = cc[1:]  # Skip background
    t = zip(cc2, sizes)
    ccc = [k[0] for k in t if (k[1] < max_area and k[1] > min_area)]
    
    # Extract patches around centers
    for point in ccc:
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round(point[1] - bbox))
        endy = starty + bbox*2
        
        if (startx > 0 and startx < img.shape[0] - bbox*2 and 
            starty > 0 and starty < img.shape[1] - bbox*2):
            
            defim.resize(defim.shape[0] + 1, bbox*2, bbox*2)
            defim[defcount] = img[startx:endx, starty:endy]
            defcoord.resize(defcoord.shape[0] + 1, 2)
            defcoord[defcount] = point[0:2]
            defcount += 1
            
    return thresh, defim, defcoord

# %% [markdown]
# Here is an example of how it works on a single frame:

# %%
thresh, defim, defcoord = extractDefectsOverlap(imgdata[fnum], classpred, softmax_threshold = 0.5, bbox=16)

# %%
fig302 = plt.figure(302, figsize=(5,5))
ax = fig302.add_axes([0, 0, 1, 1])
ax.imshow(thresh)
bbox = 16

for point in defcoord:
    startx = int(round(point[0] - bbox))
    endx = startx + bbox*2
    starty = int(round( point[1] - bbox))
    endy = starty + bbox*2
    
    p = patches.Rectangle( (starty, startx), bbox*2, bbox*2,  fill=False, edgecolor='red', lw=4)
    ax.add_patch(p)
    
ax.axis('off')

# %% [markdown]
# ### 3.3 Applying to the whole movie

# %% [markdown]
# Now that we have the functions for the single frames, we can iterate over the movie and extract all the defects for classification

# %%
mdirectory = './Figures/' + filename[:-4] + '/Movie/'
if not os.path.exists(mdirectory):
    os.makedirs(mdirectory)

# %%
bbox = 16

totdefim = np.zeros(shape=(0, bbox*2, bbox*2))
totdefcoord = np.zeros(shape=(0, 2))
totdeffnum = np.zeros(shape=(0,1))

fig303 = plt.figure(303, figsize=(5,5))
ax = fig303.add_axes([0, 0, 1, 1])


for i, img in enumerate(imgdata):
    #print(i)
    #1. use trained model to predict defecs
    classpred = predictDefects(img, model, target_size, nb_classes=2)
    
    #2. extract defects
    thresh, defim, defcoord = extractDefects(img, classpred, 0.95, bbox)
    
    #3. record the frame
    nd = len(defim)
    deffnum = np.zeros(shape=(nd,1)) + i
    
    #4. append to the array
    totdefim = np.append(totdefim, defim, 0)
    totdefcoord = np.append(totdefcoord, defcoord, 0)
    totdeffnum = np.append(totdeffnum, deffnum, 0)
    
    ax.cla()
    ax.imshow(img)
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round( point[1] - bbox))
        endy = starty + bbox*2

        p = patches.Rectangle( (starty, startx), bbox*2, bbox*2,  fill=False, edgecolor='red', lw=4)
        ax.add_patch(p)
    
    ax.axis('off')
    
    fig303.savefig(mdirectory + '_frame_' + str(i) + '.png')

plt.close(fig303)    

# %%
frames = [5, 25, 65, 85]
n = len(frames) + 1

plt.figure(figsize=(16, 8))
for i in range(1, n):
    ax = plt.subplot(2, n, i)
    i_ = frames[i-1]
    plt.imshow(imgdata[i_], cmap = 'gray')
    plt.title('Frame {0}\n'.format(i_) + 'Raw experimental data', fontsize = 10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    ax = plt.subplot(2, n, i + n)
    plt.imshow(imgdata[i_], cmap = 'gray')
    defcoord = totdefcoord[np.where(totdeffnum == i_)[0], :]
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round( point[1] - bbox))
        endy = starty + bbox*2
        p = patches.Rectangle( (starty, startx), bbox*2, bbox*2,  fill=False, edgecolor='red', lw=1)
        ax.add_patch(p)
    plt.title('Frame {0}\n'.format(i_) + 'Localized defects', fontsize = 10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

# %% [markdown]
# Because the trained network relies only on the local characteristics of the image, it can identify defects on the later stages of system evolution when the long range periodicity of the lattice is broken due to second phase evolution and displacement and rotation of unreacted WS$_2$ fragments. 

# %% [markdown]
# We can now save the data:

# %%
np.save('defim.npy', totdefim)
np.save('defcoord.npy', totdefcoord)
np.save('deffnum.npy', totdeffnum)

# %% [markdown]
# ## 4. Clustering the defects

# %%
totdefcoord = np.load('defcoord.npy')
totdeffnum = np.load('deffnum.npy')
totdefim = np.load('defim.npy')
bbox = 16

# %% [markdown]
# Now that we extracted the defects we can use gaussain mixture model method to cluster them

# %% [markdown]
# ### 4.1 Estimating number of clusters

# %% [markdown]
# We need to find an estimation of number of different types of defects we can find in the movie. This is not a simple task, as we need to select number of clusters in such a way that it is both representative of most types we are interested in (e.g. not just bright vs dark), but, at the same time, does not start splitting hairs and overwhelming us with unnecessary details.

# %%
totdef = totdefim.reshape(totdefim.shape[0], totdefim.shape[1]*totdefim.shape[2])

# %% [markdown]
# #### 4.1.1 Dendrogram

# %%
#taken from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)
    
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# %%
Z = linkage(totdef , 'ward')

# %%
fig400 = plt.figure(400, dpi=100)
max_d = 33   # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=50,
    leaf_rotation=90.,
    leaf_font_size=6.,
    show_contracted=True,
    annotate_above=30,  # useful in small plots so annotations don't overlap
    max_d=max_d
)
plt.show()

# %% [markdown]
# Dendrogram is a one way to look at clustering. As distances between clusters become smaller, it makes less sense to plit them. In this example, we can estimate about 5-7 clusters with significant distances. 

# %% [markdown]
# #### 4.1.2 PCA

# %% [markdown]
# We can use 2 types of estimation from PCA. First, we can use dimensionality reduction method: 

# %%
pca = PCA(svd_solver = 'full')

# %%
totdef.shape

# %%
pca.fit(totdef)

# %%
pca.n_components_

# %% [markdown]
# In current run, it results in 250 components with significant contribution to variation. We can look at the Scree plot to find a more reasonable cut off:

# %%
plt.figure()
plt.plot(pca.explained_variance_ratio_[1:30], marker='o',markersize=4, linestyle='-', linewidth = 0.5, color = 'black')
plt.xlabel("number of components", fontsize = 12)
plt.ylabel('explained variance', fontsize = 12)
print(pca.explained_variance_ratio_[1:30]) 

# %% [markdown]
# In this case, plot begins to flatten out after about 5-15 components

# %% [markdown]
# #### 4.1.3 BIC method

# %% [markdown]
# We can use Bayesian Information Criterion: 

# %%
totdef2 = totdef - totdef.mean(axis=0)

# %%
maxc = 20
bics = list()
for i in range(maxc-1):
    gmm = mixture.GaussianMixture(n_components=i+2, covariance_type='full')
    gmm.fit(totdef2)
    bic = gmm.bic(totdef2)
    bics.append(bic)
    print(i +2, bic)

# %%
plt.figure()
xt = np.arange(0,20,2)
plt.plot(bics, marker='o',markersize=4, linestyle='-', linewidth = 0.5, color = 'black')
plt.xlabel("number of components", fontsize = 12)
plt.ylabel("BIC", fontsize = 12)
plt.xticks(xt,fontsize = 12)
plt.yticks(fontsize = 12)

# %% [markdown]
# In this case, optimal number of clusters is 2. However, we can see some spikes in the graph at in between 5 and 15 clusters.

# %% [markdown]
# ## 4.2 Identifying types of defects

# %% [markdown]
# Although the BIC results were somewhat ambiguous, we can choose the number of clusters we are interested in to be 5. We found it to be an optimal number of components for understanding the type of defect structures present in the data. Indeed, an increase in the number of components resulted in fine (sub-) structures of the detected defects, while decrease in the number of components produced some physically meaningless structures.

# %%
nc = 5

# %% [markdown]
# #### 4.2.1 Clustering

# %% [markdown]
# We will use GMM to identify clusters and then construct an average image for each one:

# %%
gmm = mixture.GaussianMixture(n_components=nc, covariance_type='full', random_state=8000)
gmm.fit(totdef)

# %%
clusters = gmm.predict(totdef) + 1

# %%
colors = ['black', 'blue', 'red', 'cyan', 'magenta', 'green', 'darkviolet', 'gold', 'pink', 'red', 'yellow', 'maroon', 'orangered', 'lime', 'darkblue']

# %%
cla = np.ndarray(shape=(np.amax(clusters), 32,32))
fig401 = plt.figure(401, figsize=(10, 5))

rows = int(np.ceil(float(nc)/5))
cols = int(np.ceil(float(np.amax(clusters))/rows))

gs1 = gridspec.GridSpec(rows, cols)

for i in range(np.amax(clusters)):
    cl = totdefim[clusters == i + 1]
    cla[i] = np.mean(cl, axis=0)
    
    ax = fig401.add_subplot(gs1[i])
    ax.imshow(cla[i], cmap = 'gray', interpolation = 'Gaussian')
    
    ax.axis('off')
    ax.set_title('Class ' + str(i + 1) + '\nCount: ' + str(len(cl)))
    p = patches.Rectangle( (0, 0), bbox*2-1, bbox*2-1,  fill=False, color = colors[i], lw=4)
    ax.add_patch(p)

#plt.subplots_adjust(hspace=0.6, wspace=0.4)
fig401.savefig(directory + 'gmm_nc_' + str(nc) + '.png')

# %% [markdown]
# #### 4.2.2 Show clustering on the movie

# %%
mdirectory2 = './Figures/' + filename[:-4] + '/MovieClustered_reduced/'
if not os.path.exists(mdirectory2):
    os.makedirs(mdirectory2)

# %%
fig403 = plt.figure(403, figsize=(5,5))
ax = fig403.add_axes([0, 0, 1, 1])


for i, img in enumerate(imgdata):
    
    ax.cla()
    ax.imshow(img, cmap ='gray')
    
    
    defcoord = totdefcoord[np.where(totdeffnum == i)[0], :]
    clust = clusters[np.where(totdeffnum == i)[0]]
    
    for i2, point in enumerate(defcoord):
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round( point[1] - bbox))
        endy = starty + bbox*2

        
        
        p = patches.Rectangle( (starty, startx), bbox*2, bbox*2,  fill=False, edgecolor=colors[clust[i2] - 1], lw=4)
        ax.add_patch(p)
    
    ax.axis('off')
    fig403.savefig(mdirectory2 + 'frame_' + str(i) + '.png')

plt.close(fig403)

# %% [markdown]
# Finally, we should save the data:

# %%
#save the clustering data
np.save('clusters.npy', clusters)
np.save('clusters_reduced.npy', clusters)

# %% [markdown]
# ### 4.3 Distribution of defects in time

# %% [markdown]
# #### 4.3.1 Total counts

# %% [markdown]
# We are going to calculate total number of defects and their class distribution in each frame 

# %% [markdown]
# To view the trends better, we can use moving average to smooth the curves:

# %%
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# %%
nframes = len(imgdata)
totclust = np.zeros(shape = (nframes))
totbyclust = np.zeros(shape = (nc, nframes))
totbyclust2 = np.zeros(shape = (nc, nframes))
for i in range(nframes):
    clust = clusters[np.where(totdeffnum == i)[0]]
    totclust[i] = len(clust)
    
    for i2 in range(nc):
        totbyclust[i2, i] = (clust == i2 + 1).sum()    

# %%
fig430 = plt.figure(430, dpi=150)
ax = fig430.add_axes([0.1, 0.1, 0.7, 0.9])
ax.plot(moving_average(totclust, 5))
ax.set_xlim(0, 99)
ax.set_xlabel('Frame', fontsize = 14)
ax.set_ylabel('Defect Count', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=11)

# %% [markdown]
# Here we can observe a significant increase in defects starting at around frame 50. If we look a the movie, this is about when the holes start appearing. Total number of defects peaks at frame 80, due to holes in the material occupying about 25% of the image. 

# %% [markdown]
# #### 4.3.2. "Broom" graph

# %% [markdown]
# To better understand time evolution of defects, we can construct graphs tracking defects types in (x, y, t) coordinate system

# %%
def BroomLines(spo):
    lcin = np.ndarray(shape=(len(spo), 2, 3))
        
    lcin[:, 0, 2] = spo[:, 0]
    lcin[:, 1, 2] = spo[:, 0]
    lcin[:, 0, 1] = spo[:, 1]
    lcin[:, 1, 1] = spo[:, 1]
    lcin[:, 0, 0] = i
    lcin[:, 1, 0] = (i+1) 
    
    return lcin
    

# %%
fig432 = plt.figure(432, figsize=(8, 8))
ax = fig432.add_subplot(111, projection='3d')
# We will look into behavior of defects associated with 3 distorted sattes of Mo_w dopant
nc_ = [1, 2, 3, 4, 5]

for i in range(nframes):
    clust = clusters[np.where(totdeffnum == i)[0]]
    coord = totdefcoord[np.where(totdeffnum == i)[0]]

    for c, i2 in enumerate(nc_):
        spo = coord[clust==i2]
        ctest2 = BroomLines(spo)
        lc = Line3DCollection(ctest2, linewidths=1.5, colors=colors[c])    
        ax.add_collection3d(lc)
        
        
ax.set_zlim3d([0, 512])
ax.axis([0, 100, 0, 512])
        
ax.set_zlabel('X')
ax.set_ylabel('Y')
ax.set_xlabel('Frame')

# %% [markdown]
# There are three statistical behaviors: weakly moving trajectories, stronger diffusion, and “uncorrelated events”. The most well-defined trajectories are associated with Mo dopants (class 1 and class 3). Interestingly, these Mo defects show different diffusion behaviors depending on their location in the lattice and are characterized by reversible switching between two configurations (class 1 and class 3) along their trajectories. The difference in diffusion behavior may be potentially connected to complex spatial character of strain fields during the material transformation, which may impact diffusion properties

# %% [markdown]
# We can further collapse the representation to 2D. In order to show differences in time, we can project windows of defects we found. This allows us to show defects that are continous in time more clearly than randomly occuring ones.

# %%
imsum = np.zeros(shape = (imgsrc.shape[0],imgsrc.shape[1]) )

fig435 = plt.figure(435, figsize=(8, 8))
ax = fig435.add_subplot(111)

#list all the clusters for plotting 
cltoplot = [5]

coord = np.empty((0, 2))
for cl in cltoplot:
    coord_ = totdefcoord[np.where(clusters == cl)[0]]
    coord = np.concatenate((coord, coord_))
    for i2, point in enumerate(coord):
        startx = int(round(point[0] - bbox))
        endx = startx + bbox*2
        starty = int(round( point[1] - bbox))
        endy = starty + bbox*2

        imsum[starty:endy, startx:endx] = imsum[starty:endy, startx:endx] + 1

immean = imsum/len(imgdata)

imsh = ax.imshow(immean, cmap='nipy_spectral', interpolation = 'Gaussian')
plt.colorbar(imsh)
plt.ylim(0, 512)

# %% [markdown]
# We will now isolate selected trajectories for future analysis of diffusion and transition between different states

# %%
def rem_coord(co, center, dist_center):
    return [co[0] < center[0] - dist_center, co[0] > center[0] + dist_center,
            co[1] < center[1] - dist_center, co[1] > center[1] + dist_center] 

nc_ = [5]
#selecting projections with highest intensity in the analysis based on projected windows
rois = [(256, 283), (39, 123)]
flow = {}
flow_2d = {}
for i_c, center in enumerate(rois):
    cl_coord = {}
    cl_coord_all_fr = np.empty((0, 2))
    for i in range(nframes):
        clust = clusters[np.where(totdeffnum == i)[0]]
        coord = totdefcoord[np.where(totdeffnum == i)[0]]
        cl_coord_fr = OrderedDict()
        for i2 in nc_:
            spo = coord[clust==i2]
            coord_to_rem = [idx for idx, c in enumerate(spo) if any(rem_coord(c, center, 25))]
            coord_to_rem = np.array(coord_to_rem, dtype = int)
            spo = np.delete(spo, coord_to_rem, axis = 0)
            cl_coord_fr[i2] = spo
            cl_coord_all_fr = np.concatenate((cl_coord_all_fr, spo))          
        cl_coord[i] = cl_coord_fr
    flow_2d[i_c] = cl_coord_all_fr
    flow[i_c] = cl_coord

# %%
#np.save("flow_S_cl5.npy", flow)

# %% [markdown]
# ### Diffusion and switching behaviour for selected trajectories

# %% [markdown]
# We can now plot 2-d projections of selected "flows" with calculated prediction ellipses. We will also estimate a diffusion coefficient usingn a simple 2-d random walk model.

# %%
# confidence interval
nstd = 2
# conversion coefficient (pixels to nanometers) 
px_to_nm = 0.03
# conversion coefficient (frames to time(s))
frame_t = 2.25

# %% [markdown]
# We will need these functions

# %%
def diff_coef(x, y):
    ra = np.var(x)
    rb = np.var(y)
    r = ra + rb
    D = r/(4*len(x)*frame_t)
    return D
    
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

# %%
from matplotlib.patches import Ellipse

# %%
fig436 = plt.figure(436, figsize = (5, 5), dpi = 150)
ax = fig436.add_subplot(111, aspect='equal')
ax.set_xlabel('X', style = 'italic', fontsize = 14)
ax.set_ylabel('Y', style = 'italic', fontsize = 14)
for k_f in range(len(flow_2d.keys())):
    x = flow_2d[k_f][:, 0]*px_to_nm
    y = flow_2d[k_f][:, 1]*px_to_nm
    plt.plot(x, y, 'o', c = 'magenta', markersize = 14, alpha = 0.6, zorder = 1)
    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, color='black', linewidth = 1.25, ls = 'dotted', zorder = 2)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    ax.text(np.mean(x), np.mean(y), str(k_f) , fontsize=6)
    print('Diffusion coefficient for trajectory ' + str(k_f), ':', diff_coef(x, y), 'nm^2/s')    

# %% [markdown]
# We can also make a 1-d r(t) representation of the selcted "flows"

# %%
fig = plt.figure(figsize = (5, 5), dpi = 150)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)', fontsize = 16)
ax.set_ylabel('r', style = 'italic', fontsize = 16)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
n_frames = 100
for k_f in range(len(flow.keys())):
    spo = flow[k_f]
    for frame in range(n_frames):
        for idx, (k, v) in enumerate(spo[frame].items()):
            if v.shape[0] != 0: 
                v_r = np.sqrt((v[0][0]*px_to_nm)**2 + (v[0][1]*px_to_nm)**2)
                plt.plot(frame*frame_t, v_r, 'o', c=colors[k-1], markersize = 8, alpha = 0.6)  

# %%



