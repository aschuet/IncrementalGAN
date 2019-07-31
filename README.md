## Incremental GAN

This repository contains the implementation of Incremental GAN.

### Run

To run the code in the repo, please download CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and paste the images in "data/img_align_celeba/". Once that's done, you should be able to call "python main_increment.py". Check out the arguments by adding "-h".

### Evaluate

Similarly, check "eval_increment.py" for GPU settings. You will also need to provide testing epoch as an argument "--epoch &lt;num&gt;". After that you can call "python eval_increment.py".

### Acknowledgements

The files "fid_score.py" and "inception.py" are taken/modified from https://github.com/mseitzer/pytorch-fid, kudos to the author.