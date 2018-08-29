# lsgan
A replication of Least Square GAN

## Requirements

[Theano](http://deeplearning.net/software/theano/) (latest)

[neuralnet](https://github.com/justanhduc/neuralnet)

[cloudpickle](https://github.com/cloudpipe/cloudpickle/blob/master/cloudpickle)

## Usage

```
python lsgan.py path-to-a-folder-of-images (--in_size 112) (--out_size 112) (--bs 64) (--n_iters 10000000) (--gpu 0)
```

## Results

The following examples are produced @iter 117k when training with the LSUN church outdoor dataset.

![LSUN church outdoor @iter 117k](https://github.com/justanhduc/lsgan/blob/master/samples/examples.png)

## Credits

This implementation is based on this original [repo](https://github.com/xudonmao/improved_LSGAN). If you use this implementation, you must cite the following paper

```
@inproceedings{mao2017least,
  title={Least squares generative adversarial networks},
  author={Mao, Xudong and Li, Qing and Xie, Haoran and Lau, Raymond YK and Wang, Zhen and Smolley, Stephen Paul},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  pages={2813--2821},
  year={2017},
  organization={IEEE}
}
```
