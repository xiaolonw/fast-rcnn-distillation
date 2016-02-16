# Train the HHA Alexnet model from the supervision transfer initialization weights
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 1 \
  --solver scripts/scratch/solver.prototxt \
  --imdb nyud2_image_norm_2015_trainval \
  --cfg scripts/scratch/config.prototxt \
  --iters 40000 \
  2>&1 | tee scripts/scratch/train.log

