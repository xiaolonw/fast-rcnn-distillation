# Train the HHA Alexnet model from the supervision transfer initialization weights
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 0 \
  --solver output/training_demo/alexnet_hha/solver.prototxt.hha \
  --weights data/init_models/ST_vgg_to_alexnet_hha/ST_vgg_to_alexnet_hha.caffemodel \
  --imdb nyud2_hha_2015_trainval \
  --cfg output/training_demo/alexnet_hha/config.prototxt.hha \
  --iters 40000 \
  2>&1 | tee output/training_demo/alexnet_hha/train.log.hha

# Train the VGG model on the color images
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 2 \
  --solver output/training_demo/vgg_rgb/solver.prototxt.images \
  --weights data/init_models/VGG16/VGG16.v2.caffemodel \
  --imdb nyud2_images_2015_trainval \
  --cfg output/training_demo/vgg_rgb/config.prototxt.images \
  2>&1 | tee output/training_demo/vgg_rgb/train.log.images

# Train the alexnet model on the color images
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 0 \
  --solver output/training_demo/alexnet_rgb/solver.prototxt.images \
  --weights data/init_models/CaffeNet/CaffeNet.v2.caffemodel \
  --imdb nyud2_images_2015_trainval \
  --cfg output/training_demo/alexnet_rgb/config.prototxt.images \
  2>&1 | tee output/training_demo/alexnet_rgb/train.log.images

# Merge alexnet HHA and alexnet RGB models using by net surgery
mkdir output/training_demo/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval
PYTHONPATH='.' python python_utils/do_net_surgery.py \
  --out_net_def output/training_demo/alexnet_rgb_alexnet_hha/test.prototxt.images+hha \
  --net_surgery_json output/training_demo/alexnet_rgb_alexnet_hha/init_weights.json \
  --out_net_file output/training_demo/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel

# Merge alexnet HHA and VGG RGB models using by net surgery
mkdir output/training_demo/vgg_rgb_alexnet_hha/nyud2_images+hha_2015_trainval
PYTHONPATH='.' python python_utils/do_net_surgery.py \
  --out_net_def output/training_demo/vgg_rgb_alexnet_hha/test.prototxt.images+hha \
  --net_surgery_json output/training_demo/vgg_rgb_alexnet_hha/init_weights.json \
  --out_net_file output/training_demo/vgg_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel

# Testing alexnet_rgb and alexnet_hha models
model='alexnet_rgb_alexnet_hha'; tr_set='trainval'; test_set='test'; modality="images+hha";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"

# Testing vgg_rgb and alexnet_hha models
model='vgg_rgb_alexnet_hha'; tr_set='trainval'; test_set='test'; modality="images+hha";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"

# Testing alexnet_hha models
model='alexnet_hha'; tr_set='trainval'; test_set='test'; modality="hha";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_"$modality"_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"

# Testing alexnet_rgb models
model='alexnet_rgb'; tr_set='trainval'; test_set='test'; modality="images";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_"$modality"_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"

# Testing vgg_rgb models
model='vgg_rgb'; tr_set='trainval'; test_set='test'; modality="images";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_"$modality"_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"
