import os.path
from skimage import io
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
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


def ndvi(in_raster):
    return cleanData((in_raster[:,:,3] - in_raster[:,:,2])/in_raster[:,:,3] + in_raster[:,:,2])

def gndvi(in_raster):
    return cleanData((in_raster[:,:,3] - in_raster[:,:,1])/in_raster[:,:,3] + in_raster[:,:,1])

def ndre(in_raster):
    return cleanData((in_raster[:,:,4] - in_raster[:,:,2])/in_raster[:,:,4] + in_raster[:,:,2])

def ndwi(in_raster):
    return cleanData((in_raster[:,:,1] - in_raster[:,:,3])/in_raster[:,:,1] + in_raster[:,:,3])

def nrbi(in_raster):
    return cleanData((in_raster[:,:,2] - in_raster[:,:,0])/in_raster[:,:,2] + in_raster[:,:,0])

def savi(in_raster):
    return cleanData((1-0.5) + ((in_raster[:,:,3] - in_raster[:,:,2])/(in_raster[:,:,3] + in_raster[:,:,2] + + 0.5)))

def sr(in_raster):
    return cleanData(in_raster[:,:,3]/in_raster[:,:,2])

def gvi(in_raster):
    return cleanData(in_raster[:,:,3]/in_raster[:,:,1])

def totalBrightness(in_raster):
    out = np.zeros((in_raster.shape[0], in_raster.shape[1]))
    for i in range(in_raster.shape[2]):
        out += in_raster[:,:,i]
    return cleanData(out)

def cleanData(in_data):
    in_data[np.isnan(in_data)] = 0
    in_data[np.isinf(in_data)] = 0
    return in_data

def getData(in_path):
    bandRef = 'B02'
    bandsRel = {'B02': 'R10m', 'B03': 'R10m', 'B04': 'R10m', 'B08': 'R10m'}
    order = ['B02', 'B03', 'B04', 'B08']

    listOfFunctions = [ndvi, gndvi, ndwi, nrbi, savi, sr, gvi]

    dirRef = os.path.join(in_path, os.path.join(bandsRel[bandRef], bandRef))
    dirs = os.listdir(dirRef)
    X = []
    Y = []

    r1ref = io.imread(os.path.join(dirRef, dirs[0]))

    cntImgs = 0
    for dir in dirs:
        r1ref = io.imread(os.path.join(dirRef, dir))
        out = np.zeros((r1ref.shape[0], r1ref.shape[1], len(bandsRel.keys()) + len(listOfFunctions) + 1))
        cnt = 0
        for band in order:
            filePath = os.path.join(in_path, bandsRel[band], band, dir)
            im = io.imread(filePath)
            im = resize(im, (out.shape[0], out.shape[1]), anti_aliasing=True)
            out[:,:,cnt] = im
            cnt += 1

        for vi in listOfFunctions:
            out[:,:,cnt] = vi(out[:,:,0:len(order)])
            cnt += 1

        filePath = os.path.join(os.path.join(in_path, 'R10m'), 'reference', dir)
        mask = io.imread(filePath)
        mask[(mask==1) | (mask==2) | (mask==4) | (mask==5) | (mask==6)] = 2
        mask[np.bitwise_or(mask==7, mask==8)] = 1
        out[:,:,cnt] = mask
        data = out.reshape([im.shape[0]*im.shape[1],len(bandsRel.keys())+len(listOfFunctions)+1])
        data = data[data[:, -1] != 0]
        X.extend(data[:,0:-2])
        Y.extend(data[:,-1])
#        if(cntImgs==2):
#            break
        cntImgs += 1
    return X, Y


if __name__ == '__main__':
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
        RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10, n_estimators=100)

    ]

    names = [
        "Decision Tree",
        "Naive Bayes",
        "Random Forest"
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

