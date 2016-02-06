set -x
tr_set='trainval'
test_set='test'

modality="images+hha"
for model in "alexnet_rgb_alexnet_hha" "vgg_rgb_alexnet_hha"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
    --def output/$model/test.prototxt.$modality \
    --net output/$model/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
    --imdb nyud2_"$modality"_2015_"$test_set" \
    --cfg output/$model/config.prototxt
done
