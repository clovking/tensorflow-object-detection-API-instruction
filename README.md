# tensorflow-object-detection-API-instruction
> 注：文档或脚本中的一些路径需要按实际修改
1. 安装TensorFlow
2. 下载tensorflow/models模块，git clone https://github.com/tensorflow/models.git
3. 编译protobuf <br>
./research$ protoc object_detection/protos/*.proto --python_out=.  # .表示model的根目录
4. slim 文件夹加入到 PYTHONPATH环境变量中，用Slim作特征抽取 <br>
./research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 或 将 export PYTHONPATH="./research:./research/slim:$PYTHONPATH" 添加到~/.bashrc最后一行并source ~/.bashrc
5. 检测 Tensorflow Object Detection API 是否正确安装 <br>
./research$ python object_detection/builders/model_builder_test.py
6. 下载ssd_mobilenet_v1_coco_11_06_2017.tar.gz放在目录A下，该路径用于保存网络，不修改API结构
7. 下载VOC2012数据集：http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/#devkit ，解压置于路径./A/B/下
8. 准备训练集：将VOC格式数据 -> .record格式(API使用)	
  8.1 将./object_detection/dataset_tools/create_pascal_tf_record.py L165 行按如下方式修改： <br>
examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main', FLAGS.set + '.txt') <br>
原文件如下	         data_dir, year, 'ImageSets', 'Main', 'aeroplane_' + FLAGS.set + '.txt'
  8.2 在./research目录下运行脚本 pascal_to_tfrecord.sh #自己修改脚本中路径绝对路径 <br>
生成的TFRecord文件pascal_train.record, pascal_val.record在./A/B/目录中
9. 拷贝./object_detection/data/pascal_label_map.pbtxt和./object_detection/samples/configs/ssd_mobilenet_v1_coco.config至 ./A/B/AngelEye/config路径下,并对ssd_mobilenet_v1_coco.config做如下修改： 
  9.1 num_classes:修改为自己的classes num ，这里为20 
  9.2 将所有PATH_TO_BE_CONFIGURED的地方类比修改为自己之前设置的路径（5处），按照自己的路径修改即可，修改如下： <br>
        fine_tune_checkpoint: “.A/B/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt”  <br>
        input_path: ".A/B/AngelEye/config/pascal_train.record" <br>
        label_map_path: “.A/B/AngelEye/config/pascal_label_map.pbtxt” <br>
        input_path: “.A/B/AngelEye/config/pascal_val.record” <br>
        label_map_path: “.A/B/AngelEye/config/pascal_label_map.pbtxt” <br>
10. 训练: tensorflow-gpu <br>
    在model目录下运行： ./train_models.sh脚本 
11. evaluation评估
  11.1 运行脚本inferring_detections.sh <br>
        需用到冻结模型 frozen_inference_graph.pb，输出validation_detections.record
  11.2 运行脚本create_evaluation_pbtxt.sh, 自动生成两个新文件: validation_eval_config.pbtxt , validation_input_config.pbtxt
  11.3 run evaluation，运行脚本evaluation_models.sh ， 生成metrics.csv文件，得到AP, mAP等参数
12. 冻结模型, 运行脚本 frozen_inference_graph.sh
13. 测试，运行脚本 test_models.sh
14. 可视化 tensorboard --logdir A/B/ssd_mobilenet_train_logs/ , 然后在浏览器中输入 http://192.168.1.108:6006/ ，即出现可视化界面
15. 参数修改： <br>
    IOU参数：修改./research/object_detection/utils/object_detection_evaluation.py文件中Line301 class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator),def __init__(matching_iou_threshold=0.66)中数值即可(默认0.5)。 <br>
    metrics参数： 修改validation_eval_config.pbtxt文件中metrics_set: 'open_images_metrics' 或 ‘pascal_voc_metrics’
