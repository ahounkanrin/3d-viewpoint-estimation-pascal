import os
import pandas as pd
import oct2py
from oct2py import Oct2Py

oc = Oct2Py()
DATA_DIR = "/home/anicet/Datasets/PASCAL3D+_release1.1/"

classes = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car",
         "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
categories = ["train", "val"]

count = 0
for class_ in classes:
    for category in categories:
        df = pd.DataFrame()
        images = []
        labels = []  
        annotations = []
        azimuth = []
        elevation = []    
        theta = []
        bbxmin = []
        bbymin = []
        bbxmax = []
        bbymax = []
            
        with open("./{}/{}_{}.txt".format(class_, class_, category), "r") as f:
            Lines = f.readlines()
            for line in Lines:
                imgid, negorpos = line.split()
                if int(negorpos) == 1:
                    s = oc.load(DATA_DIR+"Annotations/{}_pascal/".format(class_)+imgid+".mat")
                    images.append("Images/{}_pascal/".format(class_)+imgid+".jpg")
                    annotations.append("Annotations/{}_pascal/".format(class_)+imgid+".mat")
                    labels.append("{}".format(class_))
                    #print(imgid)
                    v = s.record.objects.viewpoint
                    bb = s.record.objects.bndbox
                    if type(v) == oct2py.io.Cell:
                        for i in range(len(s.record.objects.viewpoint[0])):
                            v = s.record.objects.viewpoint[0][i]
                            if len(v) == 0:
                                continue
                            if type(v.azimuth)==float and type(v.elevation)==float and type(v.theta)==float:
                                break
                    if type(bb) == oct2py.io.Cell:
                        for i in range(len(s.record.objects.bndbox[0])):
                            bb = s.record.objects.bndbox[0][i]
                            if len(bb) == 0:
                                continue
                            if type(bb.xmin)==float and type(bb.ymin)==float and type(bb.xmax)==float and type(bb.ymax)==float:
                                break
                    
                    #print(v)
                    xmin = bb.xmin
                    ymin = bb.ymin
                    xmax = bb.xmax
                    ymax = bb.ymax
                    print(imgid, xmin, ymin, xmax, ymax)
                    assert type(bb.xmin)==float and type(bb.ymin)==float and type(bb.xmax)==float and type(bb.ymax)==float, "Wrong bounding box values"
                    az = v.azimuth
                    if not (type(az) == float or type(az) == int) :
                        az = v.azimuth_coarse
                        print("Using azimuth coarse")
                        count +=1
                    el = v.elevation
                    if not (type(el) == float or type(el) == int):
                        el = v.elevation_coarse
                        print("Using elevation coarse")
                    t = v.theta
                    if not (type(t) == float or type(t) == int):
                        t = 0.0
                        count += 1
                        print("Using theta coarse")
                    print(imgid, az, el, t)
                    azimuth.append(az)
                    elevation.append(el)
                    theta.append(t)
                    bbxmin.append(int(xmin))
                    bbymin.append(int(ymin))
                    bbxmax.append(int(xmax))
                    bbymax.append(int(ymax))

                
        df["image"] = images
        df["label"] = labels
        df["annotation"] = annotations
        df["xmin"] = bbxmin
        df["ymin"] = bbymin
        df["xmax"] = bbxmax
        df["ymax"] = bbymax
        df["azimuth"] = azimuth
        df["elevation"] = elevation
        df["theta"] = theta
        df.to_csv("./{}/{}_pascal_{}.csv".format(class_, class_, category), sep=",", index=False)
print("INFO: number of coarse labels = ", count)
