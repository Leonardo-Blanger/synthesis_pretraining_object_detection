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

(it may take several hours)

This will train all the models, and save their weights and training histories inside the `faces` directory. After this, it is possible to plot the results on the validation set during training of the full data runs with the following.

``` bash
python plot_results_valid.py
```

You should get results similar to the following:

<img src="/qrcodes/ssdmobilenet_valid_results_qrcodes.png" width=230><img src="/faces/ssdmobilenet_valid_results_faces.png" width=230><img src="/birds/ssdmobilenet_valid_results_birds.png" width=230><img src="/cars/ssdmobilenet_valid_results_cars.png" width=230>

<img src="/qrcodes/ssdresnet50_valid_results_qrcodes.png" width=230><img src="/faces/ssdresnet50_valid_results_faces.png" width=230><img src="/birds/ssdresnet50_valid_results_birds.png" width=230><img src="/cars/ssdresnet50_valid_results_cars.png" width=230>


Next, evaluate all the saved model weights on the test set with the following.

``` bash
python generate_results.py
```

This saves the test set results for all models and runs on the `results.csv` table. We can plot the graph with these results using:

``` bash
python plot_results_test.py
```

<img src="/qrcodes/ssdmobilenet_test_results_qrcodes.png" width=230><img src="/faces/ssdmobilenet_test_results_faces.png" width=230><img src="/birds/ssdmobilenet_test_results_birds.png" width=230><img src="/cars/ssdmobilenet_test_results_cars.png" width=230>

<img src="/qrcodes/ssdresnet50_test_results_qrcodes.png" width=230><img src="/faces/ssdresnet50_test_results_faces.png" width=230><img src="/birds/ssdresnet50_test_results_birds.png" width=230><img src="/cars/ssdresnet50_test_results_cars.png" width=230>


You can also print all the numeric results using:

``` bash
python generate_tables.py
```

## Ablation Experiments

### Pretraining Ablation

These set of experiments aimed at evaluating the importance of the pretraining initialization strategy followed by finetuning on real data. For this, we compared it against a single training session using real and synthesized data mixed together. For fairness, the single training session is run for the same number of iterations as the pretraining plus finetuning stages.

For running this ablation experiments, do the following:

```bash
cd faces
python pretraining_ablation.py
python generate_results_pretraining_ablation.py
```

This will train the evaluated models and save their weight files and training histories inside the `faces` directory. Then it will run each one of these models on the real Faces test set, and save the results on the `results_pretraining_ablation.csv` table. With this, you can print the values that appear on the tables like this:

```bash
python plot_results_pretraining_ablation_ablation.py
```

### Naive Pasting Ablation

These set of experiments were for checking the advantage of properly segmenting the objects. For this, we compared our proposed synthesis pipeline with a slight variation that simply pastes the full classification samples instead of using the ReDO method to segment them. These "naive" samples consider the whole classification image frame as bounding box label.

To preproduce these experiments, you first need run the naive synthesis script, which is similar to the normal synthesis script mentioned above.

```bash
cd faces
python generate_fake_data_naive.py --num_samples=200000
python naive_pasting_ablation.py
python generate_results_naive_pasting_ablation.py
```

Again, in order to print the table values, just run:

```bash
python plot_results_naive_pasting_ablation.py
```

### No GAN Ablation

These set of experiments checked wether or not the use of GAN generated classification images as the basis for synthesized samples degrades the results, in comparision to using real classification images.

In order to run these experiments, one needs to get the real image datasets that were used to train the GANs that were used to generate each type of object. That is, [FFHQ](https://github.com/NVlabs/ffhq-dataset) for Faces, [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) for Birds (which you shold already have if ran the previous experiments), and [LSUN Cars](https://www.yf.io/p/lsun) for Cars.

For the Faces, you can use the smaller resolution [FFHQ-Thumbnails](https://www.kaggle.com/greatgamedota/ffhq-face-data-set). This will be a smaller file, and these smaller images are already good enough for composing objects.

For the LSUN Cars, will will need to download the very big file, and use the [helper code](https://github.com/fyu/lsun) maintained by the dataset authors to extract the car images. It is not necessary to extract all the images. The results reported on the text consider 10000 images.

Check the respective `generate_fake_data_nogan.py` scripts for details specific to each type of object.

To reproduce these experiments, just run:

``` bash
cd faces
python generate_fake_data_nogan.py --num_samples=200000
python nogan_ablation.py
python generate_results_nogan_ablation.py
```

And to print the table values, run:

``` bash
python plot_results_nogan_ablation.py
```

