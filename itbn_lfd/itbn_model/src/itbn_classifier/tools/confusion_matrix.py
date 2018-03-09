'''
import matplotlib.pyplot as plt 
import cv2
import numpy as np

#a = np.array([[1, 0.75, 0.5], [0.75, 1, 0.75], [0.5, 0.75, 1]])

a = np.array([[0.964,	0,	0.036],
	[0.431,	0.321,	0.248],
	[0.083,	0.083,	0.833]])



a *= 255
mod = 80
a = cv2.resize(a,None,fx=mod, fy=mod, interpolation = cv2.INTER_NEAREST)

cv2.imwrite("img.png", a) 

'''


import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["font.family"] = "Times New Roman"
'''
a = np.array([[0.964,	0,	0.036],
	[0.431,	0.321,	0.248],
	[0.083,	0.083,	0.833]])
'''

a = np.array([[0.991,	0,	0.009],
	[0.009,	0.990,	0.001],
	[0.058,	0.105,	0.837]])

a = np.array([[0.858,	0.131,	0.11],
	[0.025,	0.969,	0.006],
	[0,	0.019,	0.981]])

#fig = plt.figure(figsize=(5, 5))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    print(a)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)

    plt.title(title)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
size = 7
fig = plt.figure(figsize=(size, size))
plot_confusion_matrix(a, classes=['No Action', 'Robot Action', 'Human Action'],
                      title='')

fig.savefig("img.png", bbox_inches='tight', dpi=fig.dpi)