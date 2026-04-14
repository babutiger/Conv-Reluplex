# Conv-Reluplex

This repository contains the code for the paper `Conv-Reluplex : A Verification Framework For Convolution Neural Networks`.

Legacy research code for generating and reconstructing adversarial examples on a small MNIST convolutional network by combining:

- CNN training and parameter export
- Reluplex / Leaky-Reluplex based verification on the fully connected part
- unpooling and convolution-layer constraint reconstruction
- LP solving with `pulp` to recover pixel-space adversarial images

This repository is a code-and-artifacts snapshot. It includes pretrained checkpoints, intermediate parameter dumps, generated examples, and solver logs used by the original experiments.

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

This codebase is not a modern, plug-and-play package. Before running it, expect to adapt a few legacy assumptions:

- It uses TensorFlow 1.x style APIs such as `tf.placeholder` and `tf.contrib`.
- Many files import modules from `mycode.mnist_all_minish_one_map_9_9`, which reflects the author's original local package layout.
- Some scripts contain hard-coded absolute paths, especially in `s0_parameter_all.py` and `s1_train_28_minish_one.py`.
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

1. Update `file_base` and related paths in `s0_parameter_all.py`.
2. Update MNIST dataset paths in `s1_train_28_minish_one.py`.
3. Fix or mirror the original import layout under `mycode.mnist_all_minish_one_map_9_9`.
4. Train or reuse a checkpoint from `model_9_9/` or `adversarial_train/`.
5. Run the step scripts in sequence from `s1` through `s9`.
6. Use Reluplex or Leaky-Reluplex externally between `s3` and `s4`.

## Related External Tools

- Reluplex: <https://github.com/guykatzz/ReluplexCav2017>
- Leaky-Reluplex: <https://github.com/Lzsxx/Leaky-Reluplex>

## Current Status

The repository is now versioned and published on GitHub. It is best understood as a preserved experimental code snapshot rather than a cleaned production package.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
