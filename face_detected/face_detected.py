import numpy as np
import cv2, os, math, os.path, glob, random
from ctypes import *
from sklearn.svm import LinearSVC

dll = np.ctypeslib.load_library('zmGabor', '.')  # 调用C++动态链接库
print
dll.gabor
dll.gabor.argtypes = [POINTER(c_uint8), POINTER(c_uint8), c_int32, c_int32, c_double, c_int32, c_double, c_double]


def loadImageSet(folder, sampleCount=5):
    trainData = [];
    testData = [];
    yTrain = [];
    yTest = [];
    for k in range(1, 41):
        folder2 = os.path.join(folder, 's%d' % k)
        data = [cv2.imread(d.encode('gbk'), 0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        sample = random.sample(range(10), sampleCount)
        trainData.extend([data[i] for i in range(10) if i in sample])
        testData.extend([data[i] for i in range(10) if i not in sample])
        yTest.extend([k] * (10 - sampleCount))
        yTrain.extend([k] * sampleCount)
    return trainData, testData, np.array(yTrain), np.array(yTest)


def getGaborFeature(m):
    res = []
    for i in range(6):
        for j in range(4):
            g = np.zeros(m.shape, dtype=np.uint8)
            dll.gabor(m.ctypes.data_as(POINTER(c_uint8)), g.ctypes.data_as(POINTER(c_uint8)),
                      m.shape[0], m.shape[1],
                      i * np.pi / 6, j, 2 * np.pi, np.sqrt(2))
            # res.append(cv2.dct(g[:10,:10].astype(np.float)))                            #先DCT变换再取低频系数
            # res.append(g[::10,::10])                                                    #直接子采样
            # res.append(cv2.blur(g, (10,10))[5::10, 5::10])                              #先均值滤波再子采样
            res.append(255 - cv2.erode(255 - g, np.ones((10, 10)))[5::10, 5::10])  # 先最大值滤波再子采样
    return np.array(res)


def main(folder=u'face.jpg'):
    trainImg, testImg, yTrain, yTest = loadImageSet(folder)

    xTrain = np.array([getGaborFeature(d).ravel() for d in trainImg])
    xTest = np.array([getGaborFeature(d).ravel() for d in testImg])

    lsvc = LinearSVC()  # 支持向量机方法
    lsvc.fit(xTrain, yTrain)
    lsvc_y_predict = lsvc.predict(xTest)
    print(u'支持向量机识别率: %.2f%%' % (lsvc_y_predict == np.array(yTest)).mean())


if __name__ == '__main__':
    main()