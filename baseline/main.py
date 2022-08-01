import os.path
from skimage import io
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import argparse

def parse_args():
    """Method that handles arguments
    :return parsed arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ir", "--inrasters", required=True, type=str,
                        help="Folder composed by the rasters that will be analyzed")
    args = ap.parse_args()
    return args


def getData(in_path):
    """Composes the rasters based on the sequence of spectra
    :input in_path: base path
    :return X: input data, Y: class
    """
    refResolution = 'R10m'
    pathResolution = os.path.join(in_path, refResolution)
    bands = ['B02', 'B03', 'B04', 'B08']

    dirs = os.listdir(os.path.join(pathResolution, bands[0]))
    X = []
    Y = []

    cntImgs = 0
    for dir in dirs:
        out = None
        cnt = 0
        for band in bands:
            filePath = os.path.join(pathResolution, band, dir)
            im = io.imread(filePath)
            if(out is None):
                out = np.zeros([im.shape[0], im.shape[1], len(bands)+1])
            out[:,:,cnt] = im
            cnt += 1
        filePath = os.path.join(pathResolution, 'reference', dir)
        mask = io.imread(filePath)
        mask[(mask==1) | (mask==2) | (mask==4) | (mask==5) | (mask==6)] = 2
        mask[np.bitwise_or(mask==7, mask==8)] = 1
        out[:,:,cnt] = mask
        data = out.reshape([im.shape[0]*im.shape[1],len(bands)+1])
        data = data[data[:, -1] != 0]
        X.extend(data[:,0:-2])
        Y.extend(data[:,-1])
#        if(cntImgs==2):
#            break
        cntImgs += 1
    return X, Y


if __name__ == '__main__':
    """Read the input data in the proper format, train the list of classifiers, and print stats per model.
    :input in_path: base path
    """
    args = parse_args()
    in_folder = args.inrasters

    datasetPath = in_folder
    trainPath = os.path.join(datasetPath, 'train')
    validationPath = os.path.join(datasetPath, 'validation')
    testPath = os.path.join(datasetPath, 'test')

    Xtrain, Ytrain = getData(trainPath)
    #    Xvalidation, Yvalidation = getData(trainPath)
    Xtest, Ytest = getData(testPath)

    classifiers = [
        DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, min_samples_split=10),
        GaussianNB(),
        RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10, n_estimators=100),
    ]

    names = [
        "Decision Tree",
        "Naive Bayes",
        "Random Forest",
    ]

    for name, clf in zip(names, classifiers):
        clf.fit(Xtrain, Ytrain)
        ypred = clf.predict(Xtest)
        perm = [accuracy_score(ypred, Ytest),
                f1_score(Ytest, ypred, average='macro'),
                precision_score(Ytest, ypred, average='macro'),
                recall_score( Ytest, ypred, average='macro'),
                cohen_kappa_score(Ytest, ypred),
                confusion_matrix(Ytest, ypred)

                ]
        print(name, perm)