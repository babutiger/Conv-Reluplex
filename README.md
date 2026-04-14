# Conv-Reluplex

This repository contains the code for the paper `Conv-Reluplex : A Verification Framework For Convolution Neural Networks`.

Legacy research code for generating and reconstructing adversarial examples on a small MNIST convolutional network by combining:

- CNN training and parameter export
- Reluplex / Leaky-Reluplex based verification on the fully connected part
- unpooling and convolution-layer constraint reconstruction
- LP solving with `pulp` to recover pixel-space adversarial images

This repository is a code-and-artifacts snapshot. It includes pretrained checkpoints, intermediate parameter dumps, generated examples, and solver logs used by the original experiments.

For reproducibility, this repository now includes:

- a legacy import compatibility package for `mycode.mnist_all_minish_one_map_9_9`
- repository-relative path defaults instead of hard-coded `/home/...` paths
- a `requirements.txt` file
- a setup checker at `scripts/check_setup.py`

## Paper

Paper title: `Conv-Reluplex : A Verification Framework For Convolution Neural Networks`

This codebase corresponds to the experimental pipeline used in that work.

Authors: `Jin Xu`, `Zishan Li`, `Miaomiao Zhang`, `Bowen Du`

Venue: `Proceedings of the 33rd International Conference on Software Engineering and Knowledge Engineering (SEKE 2021)`

Year: `2021`

Pages: `160-165`

DOI: `10.18293/SEKE2021-085`

PDF: <https://ksiresearch.org/seke/seke21paper/paper085.pdf>

## Citation

Plain text:

`Jin Xu, Zishan Li, Miaomiao Zhang, and Bowen Du. Conv-Reluplex : A Verification Framework For Convolution Neural Networks. In Proceedings of the 33rd International Conference on Software Engineering and Knowledge Engineering (SEKE 2021), pages 160-165, 2021. DOI: 10.18293/SEKE2021-085.`

BibTeX:

```bibtex
@inproceedings{xu2021convreluplex,
  author = {Jin Xu and Zishan Li and Miaomiao Zhang and Bowen Du},
  title = {Conv-Reluplex : A Verification Framework For Convolution Neural Networks},
  booktitle = {Proceedings of the 33rd International Conference on Software Engineering and Knowledge Engineering (SEKE 2021)},
  pages = {160--165},
  year = {2021},
  doi = {10.18293/SEKE2021-085}
}
```

## What The Project Does

The workflow is roughly:

1. Train a small CNN on MNIST.
2. Pick a correctly classified sample and export network parameters.
3. Convert the fully connected part into a format suitable for Reluplex / Leaky-Reluplex.
4. Run verification externally to obtain a satisfiable adversarial assignment in feature space.
5. Re-check the adversarial feature vector against the network.
6. Map the result back through pooling / convolution constraints.
7. Solve the resulting inequality system with `pulp`.
8. Reconstruct and save the adversarial image.

## Quick Reproduction

Recommended environment: `Python 3.7` with `TensorFlow 1.15`.

1. Create an environment and install dependencies:

```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put MNIST images in one of these two ways:

- preferred: create `data_set/mnist_data/train/` and `data_set/mnist_data/test/`
- or: keep them anywhere and point the code to them with environment variables

3. Validate the local setup:

```bash
python scripts/check_setup.py
```

4. If your MNIST directories live outside the repository, export overrides:

```bash
export CONV_RELUPLEX_MNIST_TRAIN_DIR=/path/to/mnist/train
export CONV_RELUPLEX_MNIST_TEST_DIR=/path/to/mnist/test
python scripts/check_setup.py
```

5. Run the pipeline from the repository root:

```bash
python s1_train_28_minish_one.py
python s2_predict_write_28_minish_two.py
python s3_trans_main_three.py
python s4_main_predict_from_fc_four.py
python s5_layer2_pool_to_conv_result_five.py
python s6_main_write_layer1_compute_process_six.py
python z_pulp_application/main_7_8_9_step.py
```

The repository already includes `model_9_9/`, `adversarial_train_data_*`, and
other intermediate artifacts, so you can also start from later steps if you
want to reproduce the paper results without retraining everything from scratch.

## Main Entry Files

- `s0_parameter_all.py`: central experiment configuration
- `s1_train_28_minish_one.py`: CNN training script
- `s2_predict_write_28_minish_two.py`: choose a correctly classified sample and export parameters
- `s3_trans_main_three.py`: transform parameters for Reluplex-style verification
- `s4_main_predict_from_fc_four.py`: validate the adversarial feature-space result
- `s5_layer2_pool_to_conv_result_five.py`: unpooling / intermediate reconstruction
- `s6_main_write_layer1_compute_process_six.py`: convert convolution computation into inequalities
- `z_pulp_application/s7_one_map_81_400_seven.py`: solve inequalities with `pulp`
- `z_pulp_application/s8_print_ae_eight.py`: rebuild the solved result as an image
- `z_pulp_application/s9_predict_ae_use_net_nine.py`: verify the reconstructed adversarial image

There are also two batch helpers:

- `main_2_3_batch.py`
- `main_5_6_step.py`

## Repository Layout

- `adversarial_train/`: adversarial training experiments and saved checkpoints
- `adversarial_train_data_train/`, `adversarial_train_data_test/`: generated train/test assets
- `conv_network_simulation/`: convolution-layer simulation and inequality generation
- `mnist_predict_write_example/`: saved MNIST examples and intermediate outputs
- `model_9_9/`: baseline trained model checkpoint
- `reluplex_to_ae/`: temporary adversarial feature reconstruction data
- `s2_parameter/`, `s2_parameter_divided/`: exported network parameters
- `transform_nnet_parameter/`: verification-format transformation outputs
- `z3_application/`: older Z3-based experiments
- `z_pulp_application/`: LP-based reconstruction pipeline, logs, and image outputs

## Environment Notes

This codebase is legacy research code, but the repository has been adjusted so
that it can be rerun from the repository root without mirroring the author's
original directory layout.

- It uses TensorFlow 1.x style APIs such as `tf.placeholder` and `tf.contrib`.
- Legacy imports from `mycode.mnist_all_minish_one_map_9_9` are now supported in-repo.
- Path configuration is centralized in `s0_parameter_all.py` and can be overridden with environment variables.
- External Reluplex / Leaky-Reluplex runs are not bundled here; this repository prepares data for those tools and consumes their results.

Likely Python dependencies include:

- `tensorflow` (1.x)
- `numpy`
- `scikit-image`
- `matplotlib`
- `pulp`
- `tensorboard`
- `scikit-learn`

## Minimal Setup Checklist

If you want to rerun the pipeline, start here:

1. Install dependencies from `requirements.txt`.
2. Place MNIST under `data_set/mnist_data/` or export the `CONV_RELUPLEX_MNIST_*` environment variables.
3. Run `python scripts/check_setup.py`.
4. Train or reuse a checkpoint from `model_9_9/` or `adversarial_train/`.
5. Run the step scripts in sequence from `s1` through `s9`.
6. Use Reluplex or Leaky-Reluplex externally between `s3` and `s4`.

## Related External Tools

- Reluplex: <https://github.com/guykatzz/ReluplexCav2017>
- Leaky-Reluplex: <https://github.com/Lzsxx/Leaky-Reluplex>

These external tools are referenced by the pipeline but are not vendored in
this repository. In particular, `glpk-4.60/`, `nnet/`, and upstream
Reluplex / Leaky-Reluplex source code are not redistributed here.

## Current Status

The repository is now versioned and published on GitHub. It is best understood as a preserved experimental code snapshot rather than a cleaned production package.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
