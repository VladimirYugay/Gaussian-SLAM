<p align="center">

  <h1 align="center">Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting</h1>
  <p align="center">
    <a href="https://vladimiryugay.github.io/"><strong>Vladimir Yugay</strong></a>
    ·
    <a href="https://unique1i.github.io/"><strong>Yue Li*</strong></a>
    ·
    <a href="https://staff.fnwi.uva.nl/th.gevers/"><strong>Theo Gevers</strong></a>
    ·
    <a href="https://people.inf.ethz.ch/moswald/"><strong>Martin Oswald</strong></a>
  </p>
  <div><small>*Significant contribution<small></div>
  <h3 align="center"><a href="https://vladimiryugay.github.io/gaussian_slam/index.html">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/gaussian_slam.gif" width="90%">
  </a>
</p>

## ⚙️ Setting Things Up

Clone the repo:

```
git clone https://github.com/VladimirYugay/Gaussian-SLAM
```

Make sure that gcc and g++ paths on your system are exported:

```
export CC=<gcc path>
export CXX=<g++ path>
```

To find the <i>gcc path</i> and <i>g++ path</i> on your machine you can use <i>which gcc</i>.


Then setup environment from the provided conda environment file,

```
conda env create -f environment.yml
conda activate gslam
```
We tested our code on RTX3090 and RTX A6000 GPUs respectively and Ubuntu22 and CentOS7.5.

## 🔨 Running Gaussian-SLAM

Here we elaborate on how to load the necessary data, configure Gaussian-SLAM for your use-case, debug it, and how to reproduce the results mentioned in the paper.

  <details>
  <summary><b>Downloading the Data</b></summary>
  We tested our code on Replica, TUM_RGBD, ScanNet, and ScanNet++ datasets. We also provide scripts for downloading Replica nad TUM_RGBD. <br>
  For downloading ScanNet, follow the procedure described on <a href="http://www.scan-net.org/">here</a>.<br>
  For downloading ScanNet++, follow the procedure described on <a href="https://kaldir.vc.in.tum.de/scannetpp/">here</a>.<br>
  The config files are named after the sequences that we used for our method.
  </details>

  <details>
  <summary><b>Running the code</b></summary>
  Start the system with the command:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```
  For example:
  ```
  python run_slam.py configs/Replica/room0.yaml --input_path /home/datasets/Replica/room0 --output_path output/Replica/room0
  ```  
  You can also configure input and output paths in the config yaml file.
  </details> 

  <details>
  <summary><b>Reproducing Results</b></summary>
  While we made all parts of our code deterministic, differential rasterizer of Gaussian Splatting is not. The metrics can be slightly different from run to run. In the paper we report average metrics that were computed over three seeds: 0, 1, and 2. 

  You can reproduce the results for a single scene by running:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```

  If you are running on a SLURM cluster, you can reproduce the results for all scenes in a dataset by running the script:
  ```
  ./scripts/reproduce_sbatch.sh
  ``` 
  Please note the evaluation of ```depth_L1``` metric requires reconstruction of the mesh, which in turns requires headless installation of open3d if you are running on a cluster.
  </details>

  <details>
  <summary><b>Demo</b></summary>
  We used the camera path tool in <a href="https://github.com/yzslab/gaussian-splatting-lightning">gaussian-splatting-lightning</a> repo to help make the fly-through video based on the reconstructed scenes. We thank its author for the great work.
  </details>

## 📌 Citation

If you find our paper and code useful, please cite us:

```bib
@misc{yugay2023gaussianslam,
      title={Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting}, 
      author={Vladimir Yugay and Yue Li and Theo Gevers and Martin R. Oswald},
      year={2023},
      eprint={2312.10070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
