import json
import numpy as np

## COCO imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoanalyze import COCOanalyze

import matplotlib.pyplot as plt


dataDir  = '.'
dataType = 'val2014'
annType  = 'person_keypoints'
teamName = 'Vision'

annFile  ='/home/p76094266/mmpose/data/h36m/annotation_body2d/h36m_coco_test.json' # '%s/annotations/%s_%s.json'%(dataDir, annType, dataType)
resFile  ='/home/p76094266/openmmlab/mmpose/testjson.json' # '%s/detections/%s_%s_%s_results.json'%(dataDir, teamName, annType, dataType)

print("{:10}[{}]".format('annFile:',annFile))
print("{:10}[{}]".format('resFile:',resFile))

gt_data   = json.load(open(annFile,'rb'))
imgs_info = {i['id']:{'id':i['id'] ,
                      'width':i['width'],
                      'height':i['height']}
                       for i in gt_data['images']}

team_dts = json.load(open(resFile,'rb'))
team_dts = [d for d in team_dts if d['image_id'] in imgs_info]

team_img_ids = set([d['image_id'] for d in team_dts])
print("Loaded [{}] instances in [{}] images.".format(len(team_dts),len(imgs_info)))

## load ground truth annotations
coco_gt = COCO( annFile )

## initialize COCO detections api
coco_dt   = coco_gt.loadRes( team_dts )

## initialize COCO analyze api
coco_analyze = COCOanalyze(coco_gt, coco_dt, 'keypoints')
if teamName == 'fakekeypoints100':
    imgIds  = sorted(coco_gt.getImgIds())[0:100]
    coco_analyze.cocoEval.params.imgIds = imgIds

coco_analyze.evaluate(verbose=True, makeplots=True, savedir='./test3'+'/all_plots')
coco_analyze.params.oksThrs       = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]

# set OKS threshold required to match a detection to a ground truth
coco_analyze.params.oksLocThrs    = .1

# set KS threshold limits defining jitter errors
coco_analyze.params.jitterKsThrs = [.5,.85]

# set the localization errors to analyze and in what order
# note: different order will show different progressive improvement
# to study impact of single error type, study in isolation
coco_analyze.params.err_types = ['miss','swap','inversion','jitter']

# area ranges for evaluation
# 'all' range is union of medium and large
coco_analyze.params.areaRng       = [[32 ** 2, 1e5 ** 2]] #[96 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]
coco_analyze.params.areaRngLbl    = ['all'] # 'large','medium'

coco_analyze.params.maxDets = [20]
coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
coco_analyze.summarize(makeplots=True, savedir='./test3'+'/all_plots', team_name=teamName)
# coco_analyze.summarize(makeplots=True)