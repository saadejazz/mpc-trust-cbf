from trusty import init_predictor
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":

    # fpath = "examples/smato_distracted/"
    # fpath = "rgb/"
    fpath = "examples/test1/"
    # fpath = "examples/example_images/"
    allfiles = [f for f in listdir(fpath) if isfile(join(fpath, f))]
    print(len(allfiles), "pictures found.")

    trust_estimator = init_predictor()
    # files = [
    #     "rgb/image_17_.png",
    #     "rgb/image_18_.png",
    #     # "rgb/image_19_.png",
    #     # "rgb/image_20_.png",
    #     # "rgb/image_21_.png",
    #     # "rgb/image_22_.png",
    #     # "rgb/image_23_.png",
    #     # "rgb/image_24_.png",
    #     # "rgb/image_25_.png",   
    #     ]
    files = [fpath + f for f in allfiles]

    trust_estimator.predict(files[:])