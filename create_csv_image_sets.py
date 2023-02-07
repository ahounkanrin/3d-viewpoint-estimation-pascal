import os
import pandas as pd
import pickle
from tqdm import tqdm

DATA_DIR = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"

classes = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car",
         "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
categories = ["train", "val"]
dsets = ["imagenet", "pascal"]

objId2gtbox_train =  pickle.load(open(DATA_DIR+"objId2gtbox/cate12_train.all.pkl", "rb"))
objId2gtbox_val =  pickle.load(open(DATA_DIR+"objId2gtbox/cate12_val.all.pkl", "rb"))
level = "nonOccl"
for dset in dsets:
    for class_ in tqdm(classes):
        for category in categories:
            print(dset, class_, category)
            imgIDs = []
            imgpaths = []
            labels = []
            azimuth = []
            elevation = []
            theta = []
            bbxmin = []
            bbymin = []
            bbxmax = []
            bbymax = []
            df = pd.DataFrame()
            
            data = pickle.load(open(DATA_DIR+"Catewise_obj_anno/{}_withCoarseVp.Org/{}_{}.{}.pkl".format(level, dset, category, class_), "rb")) 
            for i in range(len(data)):
                imgAnnot = data[i]
                bboxID = imgAnnot[0]
                imgid = imgAnnot[4][0]
                label = imgAnnot[1]
                if dset == "pascal" and category == "val":
                    bbox = objId2gtbox_val[bboxID]
                else:
                    bbox = objId2gtbox_train[bboxID]
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                viewpoint = imgAnnot[5]
                az = viewpoint[0]
                el = viewpoint[1]
                t = viewpoint[2]
                imgpath = "Images/{}_{}/{}".format(class_, dset, imgid)
                if dset == "pascal":
                    imgpath += ".jpg"
                else:
                    imgpath += ".JPEG"
                imgIDs.append(imgid)
                imgpaths.append(imgpath)
                labels.append(label)
                azimuth.append(az)
                elevation.append(el)
                theta.append(t)
                bbxmin.append(xmin)
                bbymin.append(ymin)
                bbxmax.append(xmax)
                bbymax.append(ymax)
            df["image_id"] = imgIDs
            df["image_path"] = imgpaths
            df["class_label"] = labels
            df["xmin"] = bbxmin
            df["ymin"] = bbymin
            df["xmax"] = bbxmax
            df["ymax"] = bbymax
            df["azimuth"] = azimuth
            df["elevation"] = elevation
            df["theta"] = theta

            
            df.to_csv("./Image_sets/{}/{}/{}_{}_{}.csv".format(class_, level, level, dset, category), sep=",", index=False)
