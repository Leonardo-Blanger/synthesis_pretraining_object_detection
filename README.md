# Sample Synthesis based Network Pretraining for Object Detection

The directories `qrcodes`, `faces`, `birds`, and `cars`, each contain scripts to run the respective experiments.

For each type of object, download the respective real world dataset as indicated in the table bellow, and extract their contents into a `data` directory on the project root.

**object** | **dataset**
---------- | -----------
QR Codes   | [QR Codes dataset](https://github.com/ImageU/QR_codes_dataset)
Faces      | [FDDB](http://vis-www.cs.umass.edu/fddb/)
Birds      | [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
Cars       | [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Generating Synthesized Data

Although the Synthesis Pipeline conceptually allows an infinite stream of samples, in practice running the image synthesis at the same time as we train a model ends up being too computationally expensive. As a workaround, we provide scripts to synthesize a very large quantity of samples ("infinite" for practical purposes).

In a similar way, we also do not run the generative model at the beginning of the Synthesis Pipeline for every instance of an object during synthesis. Instead, we generate a very large amound of fake object images, and sample from these during synthesis.

Before synthesizing samples, you first need to generate fake classification level images. The way we did this was with the publicly available [StyleGAN](https://github.com/NVlabs/stylegan) Faces and Cars, and the [DM-GAN](https://github.com/MinfengZhu/DM-GAN) model for Birds.

In order to generate samples for a particular object type, use the `generate_fake_data.py` scripts in the respective directory. For instance:

```bash
cd faces
python generate_fake_data.py --num_samples=200000
```

Please, check each of these scripts, as they provide an example on how to structure the above mentioned gathered fake images. By default, they will generate the synthesized detection samples and save them in `$PROJECT_ROOT/data/{qr_codes,faces,birds,cars}_fake/`, with labels in the format expected by the respective data loader scripts.

## Experiments

We provide scripts inside the object directories to reproduce the reported experiments. For instance, to reproduce the main experiments on faces, run:

```bash
cd faces
python run_training.py
```

### Ablations...
