cd ../

python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 1e-11 --random-seed 0 --model-name S0 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 2e-11 --random-seed 0 --model-name S0 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 5e-11 --random-seed 0 --model-name S0 --model-stamp 1110

python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 1e-11 --random-seed 1 --model-name S1 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 2e-11 --random-seed 1 --model-name S1 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 5e-11 --random-seed 1 --model-name S1 --model-stamp 1110

python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 1e-11 --random-seed 2 --model-name S2 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 2e-11 --random-seed 2 --model-name S2 --model-stamp 1110
python3 cifar_bqn_train.py --dataset CIFAR10 --model-type VGGLike_v2 --lamb 5e-11 --random-seed 2 --model-name S2 --model-stamp 1110