# Google-Colab-custom-object-detection-training-and-downloading-frozen-inference-graph
Making dataset
1. With an appropriate number of photos (my example have 20 photos of Dell laptop), I created the annotations. The tool I used is LabelImg. For the sake of simplicity, I identified a single object class, dell. It's possible to extend it to obtain models that perform object detection on multiple object classes.
2. I renamed the image files in the format objectclass_id.jpg (i.e. dell_001.jpg, dell_002.jpg). Then in LabelImg, I defined the bounding box where the object is located, and I saved annotations in Pascal Voc format.
3. Finally, I uploaded annotations files in my Google Drive account, using a single zip file with the following structure:
.zip file
|-images directory
  |-image files (filename format: objectclass_id.jpg)
|-annotations directory
  |-xmls directory
    |-annotations files (filename format: objectclass_id.xml)
4. use image_resize.py to resize the images
    
