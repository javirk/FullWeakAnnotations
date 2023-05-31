# Full or Weak annotations?

This is the repository for the paper "Full or Weak annotations? An adaptive strategy for budget-constrained annotation 
campaigns", presented at CVPR2023. It contains the code to reproduce the experiments presented in the paper.

## Links
- [Paper](https://arxiv.org/abs/2303.11678) 
- [Project page](https://javiergamazo.com/full_weak/)

## How to run
First of all, you need a surface of a dataset. A sample surface has been stored in `surfaces/sample_surface.txt`.
The format is:

    run_name, classification share (%), segmentation share (%), IoU, Dice

Then, you will have to create a `gp_config` file. You can use any `gp_config` file in the `gp_configs` folder as a template.
Remember to change `surface_file` parameter to the filename of your surface file.

Finally, you can run the method with the file `gp.py`as follows:
    
        python gp.py --config gp_configs/gp_config.txt
