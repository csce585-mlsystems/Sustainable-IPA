# Sustainable-IPA
Considering energy consumption in [IPA](https://github.com/reconfigurable-ml-pipeline/ipa). towards Sustainable AI.

## Proposal
### Problem
Sustainable AI is a subfield dedicated to considering energy usage in AI and machine learning models. Machine learning models consume a large amount of energy, but sustainability is often an afterthought for developers of these models. Additionally, for many models inference can have more of an energy impact than training, especially when large quantities of users are being served.

We are interested in adding a generalizable energy consumption metric to the [Inference Pipeline Adaptation system](https://github.com/reconfigurable-ml-pipeline/ipa) so that when developers apply this framework to their machine learning pipeline they can consider energy consumption in their selection of model variants. IPA currently considers the metrics of accuracy, cost, and latency when choosing an ideal model, but does not consider energy consumption. We designed an experiment to evaluate the impact of energy consumption on the IPA configuration search space. We will apply IPA to the Video pipeline, which is provided with the IPA code, and measure the energy consumption for every adaptation configuration in the search space.

### Reading
We will consider the following works as we research this problem:
* [CLOVER: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service](https://arxiv.org/pdf/2304.09781)
* [Sponge: Inference Serving with Dynamic SLOs Using In-Place Vertical Scaling](https://arxiv.org/pdf/2404.00704)
* [Toward Sustainable GenAI using Generation Directives for Carbon-Friendly Large Language Model Inference](https://arxiv.org/pdf/2403.12900)
* [MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant Systems for Machine Learning](https://arxiv.org/pdf/2207.11428)
* [Chameleon Cloud Documentation: Energy and Power Consumption Measurement with etrace2](https://chameleoncloud.readthedocs.io/en/latest/technical/metrics.html#energy-and-power-consumption-measurement-with-etrace2)
* [Chameleon Cloud: Bare Metal Experiment Pattern](https://developer.nvidia.com/nvidia-power-capture-analysis-tool)
* [NVIDIA Power Capture Analysis Tool](https://developer.nvidia.com/nvidia-power-capture-analysis-tool)
* [Tutorial for Demo for Open Source Summit NA 2023](https://github.com/wangchen615/OSSNA23Demo)
* [5 Steps to Deploy Cloud Native Sustainable Foundation AI Models](https://docs.google.com/presentation/d/187KrP5JIh6m9-5nD-pIiHkv7Tl0xznBg/edit#slide=id.p10)

### Data
We are using the [Video pipeline](https://github.com/reconfigurable-ml-pipeline/ipa/tree/e1f08dde84e2bb721b2c78ad7ef651134abf5380/pipelines/mlserver-final/video), which is an experimental pipeline downloadable with IPA that uses a YOLOv5 model for cropping the image and a ResNet model for classifying humans. There is a Video pipeline simulation configuration scriptin `/ipa/data/configs/pipeline-simulation` that we will use. Since we are evaluating this pipeline for energy consumption, we do not need to use any additional data for training besides what the model has already been trained on.

For testing, we will use test images like the ones used in the examples in IPA.

### Method
Since we are expanding IPA, we will start by installing the code on our Chameleon Cloud workspace and running it on the Video pipeline. We will then view existing metrics for every configuration in the configuration search space. We will run each of these configurations and measure the energy consumption for inference of a test image. We will then view the energy consumption results and analyze how they affect the other metrics.

We will measure energy consumption using Chameleon Cloud’s etrace2 tool and NVIDIA’s Power Capture Analysis Tool (PCAT). The etrace2 tool will allow us to read energy consumption over a time interval.

### Evaluate
We will evaluate these results by first determining if energy consumption is tightly coupled with any existing metric in IPA. We assume that it may be consistent with cost.

If energy consumption does provide some additional information in the search space, we will then aim to add this metric into IPA for future developers to use.

If energy consumption does not provide additional information, we will ask what other ways we can consider the importance of energy consumption in IPA. This may include GPU time-slicing, considering time of day that the model is run or where resources are located, or simply providing energy consumption data as many machine learning pipelines do not evaluate energy consumption and there is no standard open source tool for providing these metrics.

## Evaluation Experiment Design
IPA does not consider energy consumption as part of its pipeline evaluation. We want to determine if energy consumption is a relevant metric for pipeline evaluation. We are considering a lower energy consumption to be more desirable, since energy efficiency is better for the environment, with the ultimate aim of creating a more sustainable version of IPA that considers environmental impact. We have designed an experiment to analyze the energy consumption of different adaptations in the adaptation space created by IPA on a single pipeline.

![Figure 1: Video pipeline](https://github.com/csce585-mlsystems/Sustainable-IPA/blob/main/documentation/images/pipelines.png "Figure 1")

First, we shall create a script to output a plot visualizing the accuracy, cost, and latency of all adaptations in the adaptation space for the Video pipeline at the same timestep. Figure 1 above shows the pipelines represented in IPA. The video pipeline is shown at pipeline *a*. This script will be adapted from [ipa/experiements/runner/notebooks/paper-fig13-gurobi-decision-latency.ipynb](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/experiments/runner/notebooks/paper-fig13-gurobi-decision-latency.ipynb), the script used to create Figure 13 in the paper. Like in this script, our script will load the experiment data from the video pipeline using the Adaptation Parser in [ipa/experiments/utils.parser.py](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/experiments/utils/parser.py). We will then use the adaptation logs to plot every adaptation. The adaptation search space will be different combinations of model variants and batch sizes.

In our output plot, cost will be plotted on the y-axis and accuracy will be plotted on the x-axis. The points representing an adaptation in the search space will be colored based on the percentage of latency SLAs that were met, particularly for the end-to-end latency.

We will then run every adaptation in our search space over the Video pipeline data at the same timestep and measure energy consumption. We will measure energy consumption using Chameleon Cloud’s [Bare Metal Experiement Pattern](https://chameleoncloud.org/experiment/share/50692573-4094-466c-b4fe-0ed3471f8993).

We will use hardware in the Chameleon UC datacenter, specifically a `compute_cascadelake_r_ib` node recommended by [IPA](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/infrastructure/automated.md). The Chameleon [Hardware Discovery page](https://www.chameleoncloud.org/hardware/) shows these machines to have x86_64 processors with 2 CPUs, 96 threads, and 192GiB of RAM. [The processors are Intel(R) Xeon(R) Gold 6240R CPU @2.40GHz](https://chameleoncloud.org/hardware/node/sites/uc/clusters/chameleon/nodes/89e48f7e-d04f-4455-b093-2a4318fb7026/). Note that this hardware does not use a GPU. Future work could be done to deploy IPA to a GPU for faster processing. This will affect energy consumption and possibly the energy consumption measurement method.

We will then use our plotting script to add a z-axis which will represent the energy consumption of the adaptation in the adaptation space. We will analyze this new plot to find how the ideal adaptations are changed when energy consumption is evaluated as a relevant metric. Figure 2 below is a drawing of how we anticipate to visualize our results before and after including the energy consumption measurement.

![Figure 2: Experiment result visualization](https://github.com/csce585-mlsystems/Sustainable-IPA/blob/main/documentation/images/experiment_result_visualization.png "Figure 2")
