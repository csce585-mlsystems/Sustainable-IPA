# Sustainable-IPA
Considering energy consumption in [IPA](https://github.com/reconfigurable-ml-pipeline/ipa) towards Sustainable AI. Deliverables and documentation are tracked here, while code changes are tracked on the [Sustainable-IPA Fork](https://github.com/csce585-mlsystems/Sustainable-IPA-Fork).

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
We are using the [Video pipeline](https://github.com/reconfigurable-ml-pipeline/ipa/tree/e1f08dde84e2bb721b2c78ad7ef651134abf5380/pipelines/mlserver-final/video), which is an experimental pipeline downloadable with IPA that uses a YOLOv5 model for cropping the image to detected objects and a ResNet model for classifying humans. There is a Video pipeline simulation configuration scriptin `/ipa/data/configs/pipeline-simulation` that we will use. We will not be training either model, so we do not require any training data.

We shall create a test dataset for our experiement purposes. This dataset shall be 20 JPG images that contain people, as well as various objects. We will hand-select images from the opensource [COCO dataset](https://cocodataset.org/#overview). This datset makes it easy to select images with people as well as other objects of choice. The dataset also includes the ground-truth JSON files. We will use these files to generate an accuracy metric of each model in the machine learning system. We will use the entire dataset for every adaptation in the adaptation space. The dataset may be processed in different batch sizes as is configured by IPA.

### Method
Since we are expanding IPA, we will start by installing the code on our Chameleon Cloud workspace and running it on the Video pipeline. We will then view existing metrics for every configuration in the configuration search space. We will run each of these configurations and measure the energy consumption for inference of a test image. We will then view the energy consumption results and analyze how they affect the other metrics.

We will measure energy consumption using Chameleon Cloud’s etrace2 tool and NVIDIA’s Power Capture Analysis Tool (PCAT). The etrace2 tool will allow us to read energy consumption over a time interval.

### Evaluate
We will evaluate these results by first determining if energy consumption is tightly coupled with any existing metric in IPA. Energy efficiency is related to machine learning systems in many ways. We predict this metric is influenced by--and influences--every other metric in our trade-off space.

A system will generally require more compute if the machine learning models in the system are larger in terms of hyperparameters or are processing more data at each model inference, using higher batches, or inferencing at a higher frequency. This compute will be pulled from the proessor running the model, which will require more energy to power. More powerful processors with greater parallel processing capabilities may be added to balance other metrics. Therefore, we predict higher energy consumption as the compute cost increases.

A larger machine learning model may also be more accurate. Therefore, as more compute allows for more accurate models, we expect accuracy will also be correlated with energy consumption in the same way. Additionally, larger (more accurate) models may have a higher latency because the inference time will increase. However, more time-efficient processors can re-balance the latency.

The existing trade-off space in IPA is highly interconnected, and energy consumption is another metric that we predict will change as each other metric in the trade-off space changes, depending on the pipeline that IPA is applied to. We are motivated to see exactly how the energy consumption changes as other metrics change within the IPA trade-off space for a machine learning system. It is possible that energy consumption follows other metric(s) so closely that a measurement of energy consumption does not provide a much different perspective to the trade-off space. However, it is still relevant information to track, as a user of IPA may choose to compromise on cost, accuracy, or latency to meet energy efficiency SLAs.

We shall analyze the role of energy consumption in a machine learning system and propose future work accordingly. If energy consumption plays a unique role in the trade-off space, we shall recommend adding energy consumption as a metric in IPA. If it is too tightly-coupled with other metrics, our results can still be used to inform IPA user choices. An energy consumption measurement mechanism may still be useful if the user has sustainability-conscious SLAs. We may also consider other ways to affect energy consumption of machine learning systems using IPA, including GPU time-slicing, considering the time of day that the model is run or considering where compute resources are located. We also recognize that simply providing a method to measure energy consumption from within IPA is useful, as there is currently no standard open source tool for providing these metrics.

## Evaluation Experiment Design
IPA does not consider energy consumption as part of its pipeline evaluation. We want to determine if energy consumption is a relevant metric for pipeline evaluation. We are considering a lower energy consumption to be more desirable, since energy efficiency is better for the environment, with the ultimate aim of creating a more sustainable version of IPA that considers environmental impact. We have designed an experiment to analyze the energy consumption of different adaptations in the adaptation space created by IPA on a single pipeline.

![Figure 1: Video pipeline](https://github.com/csce585-mlsystems/Sustainable-IPA/blob/main/documentation/images/pipelines.png "Figure 1")

First, we shall create a script to output a plot visualizing the accuracy, cost, and latency of the selected adaptations in the adaptation space for the Video pipeline at the same timestep. Figure 1 above shows the pipelines represented in IPA. The video pipeline is shown at pipeline *a*. Our script will be adapted from [ipa/experiements/runner/notebooks/paper-fig13-gurobi-decision-latency.ipynb](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/experiments/runner/notebooks/paper-fig13-gurobi-decision-latency.ipynb), the script used to create Figure 13 in the paper. Like in this script, our script will load the experiment data from the video pipeline using the Adaptation Parser in [ipa/experiments/utils.parser.py](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/experiments/utils/parser.py). We will then use the adaptation logs to plot each adaptation.

Ten adaptations shall be selected for evaluation. These selected adaptations will cover a wide range of model variants and batch sizes. It will include pipelines that are highly accurate but slow and costly, as well as pipelines that are lightweight. The adaptions we select shall have a large variance in effort to see more of the trade-off space.

In our output plot, cost will be plotted on the y-axis and accuracy will be plotted on the x-axis. The points representing an adaptation in the search space will be colored based on the percentage of latency SLAs that were met, particularly for the end-to-end latency.

We will then run every adaptation in our search space over the Video pipeline data at the same timestep and measure energy consumption. We will measure energy consumption using Chameleon Cloud’s [Bare Metal Experiement Pattern](https://chameleoncloud.org/experiment/share/50692573-4094-466c-b4fe-0ed3471f8993). We shall run the entire experiment for three trials and average the results to increase the signal-to-noise ratio of the energy consumption measurment. This averaging will also reduce the impact of extraneous processes running on the processor on our results. These averaged results will be used for evaluation.

We will use hardware in the Chameleon UC datacenter, specifically a `compute_cascadelake_r_ib` node recommended by [IPA](https://github.com/reconfigurable-ml-pipeline/ipa/blob/e1f08dde84e2bb721b2c78ad7ef651134abf5380/infrastructure/automated.md). The Chameleon [Hardware Discovery page](https://www.chameleoncloud.org/hardware/) shows these machines to have x86_64 processors with 2 CPUs, 96 threads, and 192GiB of RAM. [The processors are Intel(R) Xeon(R) Gold 6240R CPU @2.40GHz](https://chameleoncloud.org/hardware/node/sites/uc/clusters/chameleon/nodes/89e48f7e-d04f-4455-b093-2a4318fb7026/). Note that this hardware does not use a GPU. Future work could be done to deploy IPA to a GPU for faster processing. This will affect energy consumption and possibly the energy consumption measurement method.

We will then use our plotting script to add a z-axis which will represent the energy consumption of the adaptation in the adaptation space. We will analyze this new plot to find how the ideal adaptations are changed when energy consumption is evaluated as a relevant metric. Figure 2 below is a drawing of how we anticipate to visualize our results before and after including the energy consumption measurement.

![Figure 2: Experiment result visualization](https://github.com/csce585-mlsystems/Sustainable-IPA/blob/main/documentation/images/experiment_result_visualization.png "Figure 2")

Additionally, we shall plot the relationship between existing performance metrics (cost, latency, and accuracy) and energy consumption. We shall create three different plots, each with energy consumption on the y-axis, and with the an existing metric on each x-axis. These plots will allow us to inspect the trend of the existing metrics as energy consumption is changed.

Lastly, we shall run the same experiment over different hardware setups and compare the results. We can use Chameleon Cloud hardware with different specifications.

## Future Work
We describe here our future work that we consider relevant but may be out of the scope of our current project.

### Sensitivity Analysis
For certain adaptations in the adaptation space, there may be a relatively small trade-off in one or more existing metrics for a relatively large gain in energy consumption. Although this secondary adaptation may not technically be the best choice, it may be useful to provide this information to the user and allow them to weigh energy consumption as a higher relevance for especially energy-conscious systems. Our experiment results can be used to conduct a sensitivity analysis that explores the feasibility of this feature.

### Experimenting with other Pipeline Types
IPA can be used for a wide variety of machine learning systems, including pipelines that process written language, auditory data, and other signal types. The methods we are considering for measuring energy consumption would be generalizable to other machine learning pipeline types. We could reproduce this same experiment with a different pipeline by simply changing the pipeline itself and the test dataset. We expect that energy consumption will change as the pipeline type changes, as the size of the data type likely affects how much energy is needed. Smaller data types, such as text, will likely consume less energy.

### GPU Processing
Machine learning models are considerable more efficient when deployed to the GPU because they are well-suited to parallel processing. IPA does not currently take advantage of the benefits of a GPU for it's machine learning models. We expect that deploying models in the machine learning systems used by IPA will greatly change every metric in the trade-off space. A framework like [PyTorch](https://pytorch.org/) can be used to deploy machine learning models to the GPU.

#### TPU Processing, Specialized Processors
Tensor Processing Units (TPUs) are a new type of processor developed by Google specifically for machine learning models. Field Programmable Gate Arrays (FPGAs) are integrated circuits with programmable logic gates. They can be used to develop solutions to specific machine learning applications. Additionally, TensorRT is NVIDIA's SDK for optimizing machine learning models to specific hardware with quantization and other methods. Any of these methods could be integrated with IPA in the future.

### Real-Time Energy Consumption Tracking
It may be beneficial to track changes in energy consumption in real-time, allowing developers to adjust their pipelines as the energy consumption changes. Energy consumption may even change for the same resource load, as it can be affected by time of day and physical location. If energy consumption suddenly rises or drops, a new configuration may be considered more optimized at that time.

### Expanding Energy Consumption Measurement Methods
Measuring energy consumption can be difficult. We may look into other methods of measuring energy consumption. This could include collaboration with cloud providers that already have accurate energy consumption measurment technology.

### Carbon-Awareness
Carbon-aware AI is a movement to consider the carbon footprint of machine learning systems.
IPA could select configurations based on carbon footprint SLAs, much like we are proposing for energy consumption SLAs. For carbon footprint, IPA would need to consider the energy source being used, as well as it's physical location.

#### Carbon Footprint Calculators
Energy consumption measurements may be vague or difficult to interpret. These measurements could be displayed to the developer in the form of a carbon footprint calculator, which is a more intuitive way to explain energy consumption. This information could easily be used by the developer to provide for their end-user.
