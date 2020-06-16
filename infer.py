"""
    Folder to infer stuff on
"""
import utils
import feature
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import glob

def main(triplet_model, INFER_DIR=None, IMG_DIMENSION=None, FEATURES=None, OUTPUT=None, **kwargs):
    assert isinstance(INFER_DIR, str)
    INFER_DIR += '*'

    imList = sorted(glob.glob(INFER_DIR),key=utils.numericalSort)
    print (imList[0])
    print (len(imList))

    p = 8#multiprocessing cpu_count
    pool = Pool(processes=8)
    print ("Processors Available = " + str(p))

    lst = feature.prepareFeatures(triplet_model,imList,IMG_DIMENSION,FEATURES)
    lst = np.asarray(lst)

    activations = lst


    # """
    #     Hardcoded stuff for multi core :p
    # """
    lst = pool.map(feature.featureSelect,((activations,imList,1),(activations,imList,2),
                                  (activations,imList,3),(activations,imList,4),(activations,imList,5),
                                  (activations,imList,6),(activations,imList,7),(activations,imList,8)))

    tP = 0
    masterBlue = []
    masterRed = []
    masterVar = np.zeros(shape = (FEATURES*3))
    for x in range(len(lst)):
        for y in range(len(lst[x][0])):
            masterBlue.append(lst[x][0][y])
        for z in range(len(lst[x][1])):
            masterRed.append(lst[x][1][z])
    for x in range(len(lst)):
            tP+= lst[x][2]
    print ("Total true positives = " + str(tP))
    masterRed = np.sort(masterRed)
    plt.hist(masterBlue, density=True, bins=120, histtype='stepfilled', color='b',alpha=0.7, label='Same')
    plt.hist(masterRed, density=True, bins=120, histtype='stepfilled', color='r', label='Different')
    #plt.show()
    plt.savefig(OUTPUT)

    # print ("Standard deviation same = " + str(np.std(masterBlue)))
    # print ("Standard deviation different = " + str(np.std(masterRed)))
    # print ("Blah = " + str(len(masterBlue)))
    # print ("Min Same class = " + str(min(masterBlue)))
    # print ("Min Different class = " + str(min(masterRed)))
    # print ("Minimum = " + str(masterRed[:60]))
    # print ("Max = " + str(masterRed[-60:]))

    # print ("Average Same class = " + str(sum(masterBlue) / float(len(masterBlue))))
    # print ("Average Different class = " + str(sum(masterRed) / float(len(masterRed))))


    # u = np.mean(masterRed)
    # entropy = (u*(1-u))/np.var(masterRed)
    # print ("Entropy = " + str(entropy))

    # blueFile = open('blue.txt', 'w')
    # for item in masterBlue:
    #     blueFile.write("%s\n" % item)

    # redFile = open('red.txt', 'w')
    # for item in masterRed:
    #     redFile.write("%s\n" % item)


