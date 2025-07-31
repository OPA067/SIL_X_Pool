#### Training
```python MSRVTT
python train.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --log_step=100 --evals_per_epoch=1 --batch_size=25 --videos_dir=MSRVTT/videos/ --DSL
```

```python DiDeMo
python train.py --exp_name=DiDeMo-train --dataset_name=DiDeMo --log_step=10 --evals_per_epoch=1 --batch_size=8 --videos_dir=DiDeMo/videos/ --num_frames=24
```

```python Charades
python train.py --exp_name=Charades-train --dataset_name=Charades --log_step=10 --evals_per_epoch=1 --batch_size=25 --videos_dir=Charades/videos/
```

```python ActivityNet
python train.py --exp_name=ActivityNet-train --dataset_name=ActivityNet --log_step=10 --evals_per_epoch=1 --batch_size=25 --videos_dir=ActivityNet/videos/
```

#### Testing
```python MSRVTT
python test.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --batch_size=25 --videos_dir=MSRVTT/videos/
```