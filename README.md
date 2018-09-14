# AVSD Baseline

Modified baseline for DSTC7 track 3

original: https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge/tree/master/AVSD_Baseline/Baseline
## Environment Setup
1. install requirement python packages

   `pip install -r requirements.txt`
   
2. install pytorch depends on your environment

3. install cocotools for python

   `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`

4. install Stanford NLPCore for SPICE Metric

   `./utils/get_stanford_models.sh`

## How to get the data
Video data: CHARADES for human action recognition datasets

https://allenai.org/plato/charades/

Prototype dataset: 6172(training), 732(validation), 733(test) https://drive.google.com/drive/u/2/folders/1JGE4eeelA0QBA7BwYvj89kSClE3f9k65

The text file in prototype dataset organized in the following json format:
```
{
 "type":str
 "version":str
 "dialogs":[{
            "image_id":str
            "caption":str
            "dialog":[{
                     "answer":str, 
                     "question":str
                     },...]
            "summary":str
            },...]
}
```
## How to run the baseline (use prototype dataset)
1. unzip compressed dataset
2. modified `$dataset` in `run.sh` to the path in step 1
3. kick off training process

   `sh run.sh`
