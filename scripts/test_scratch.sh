set -x
tr_set='trainval'
test_set='test'

modality="image_norm"
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def scripts/scratch/test.prototxt \
    --net /nfs.yoda/xiaolonw/fast_rcnn/models/scratch/fast_rcnn_iter_40000.caffemodel \
    --imdb nyud2_"$modality"_2015_"$test_set" \
    --cfg scripts/scratch/config.prototxt
