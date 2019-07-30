## Incremental GAN

This repository contains the implementation of Incremental GAN.

### Run

To run the code in the repo, please download CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and paste the images in "data/img_align_celeba/". You can edit GPU settings in "main_increment.py". Once that's done, you should be able to call "python main_increment.py"

### Evaluate

Similarly, check "eval_increment.py" for GPU settings. You will also need to provide testing epoch in the code. After that you can call "python eval_increment.py".

### Acknowledgements

The files "fid_score.py" and "inception.py" are taken/modified from https://github.com/mseitzer/pytorch-fid.