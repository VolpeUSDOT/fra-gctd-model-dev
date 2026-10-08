# FRA Grade Crossing Trespass Detection Machine Learning Model Development
This repository is home to the python scripts used to modify and fine-tune the pre-trained detection model.

#### Folder Structure
**/PhaseOne**   *Phase one model development using Tensorflow 1.x*

**/PhaseTwo**   *Phase two model development using PyTorch 1.x*

## Requirements

- Python 3.x (Tested on 3.7.x)
- Tensorflow 1.15.x (Phase One)
- PyTorch >=1.5.x (Phase Two)

## Dataset Downloads

The Phase One Flowers and CIFAR-10 converters require certificate-verified HTTPS
downloads and pinned SHA-256 digests. Archives are extracted into private staging
directories using a file-and-directory-only policy compatible with Python 3.7+;
links, special files, unsafe paths, and oversized archives are rejected.

An existing `flower_photos` or `cifar-10-batches-py` directory is not overwritten
or merged. Use a fresh dataset output directory when retrying an interrupted
conversion. Temporary downloads and rejected extraction data are cleaned up;
training and conversion still require the legacy dependencies above.

The digest pins come from TensorFlow Datasets `tf_flowers/checksums.tsv` at
commit `e65dc78511d1e551e382928e0f54836cd219172f` and Keras
`keras/src/datasets/cifar10.py` at commit
`9bcdbc786a8948cce506b9c5fcef2ce33e6efd07`. Update these pins only after
independently verifying replacement archives against a trusted publication.

Run the download security regression tests from `PhaseOne` without TensorFlow:

```sh
python -m unittest datasets.dataset_download_test datasets.dataset_conversion_security_test -v
```

## License

MIT
