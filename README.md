# COMS6998_FinalProj: Building a Super Model: Pruning Techniques and Distributed DL for Training Efficiency

**A description of the project**  <br>
In this project we set out to optimize convolution neural networks through exploring the techniques of data augmentation, pruning, mixed precision, quantization and synchronous training. 



**A description of the repository** <br>
This repository contains the codes and all relevant training logs/models produced during the project. <br>
The training logs and models for ResNet 20 and ResNet 44 are stored in separate folders respectively.<br>

Under the folder codes, we include all training codes for the trials of exploration, for example, testing constant sparsity, poly-sparsity, and distributed training, using CIFAR-10 and CIFAR-100 datasets. <br>

The notebooks that produced the results for our super model trials are named super_sixty.ipynb and super_seventy.ipynb in the codes folder. <br>

We tried two different “super”-models, one with final sparsity 60% and one with 70% sparsity. As mentioned in the presentation, the hyperparameters are found through 29 trial runs on a purely pruning based model. We selected the maximal sparsity, 60%, that gives the desired validation accuracy. We tried both 60% and 70% in order to fine tune the model. The training log and trained models are stored under /Super_Model/super_0.6 and  /Super_Model/super_0.7 <br>


**Example commands to execute the code**   <br>     
We included all codes as jupyter notebooks, and all the notebooks can be run on GCP VMs. Note that for the distributed training of the supermodel, it needs to be run on Vertex AI. The set up of Vertex AI is described in detail in this video: https://www.youtube.com/watch?v=rAGauhXYgw4&list=WL&index=1 . When logged into Vertex AI workbench, press the "JupyterLab" button to launch jupyterlab, and upload the jupyter notebook using the UI. Then, on the upper-right corner, select the machine configuration. We used 4 CPUs, 15 GB RAM, and 1 Tesla V100. Now, the distributed training notebook can be run just as a standard jupyter notebook. Run super_demo.ipynb to train a Cifar100 super model. You must first change the logname in resnet_training() to your appropriate file path to store training data. Feel free to experiment with hyperparameters such as inital/final sparsity, pruning frequency, pruning schedule, etc as instructed in the demo notebook. Another notebook, mixed_precision.ipynb is also included which demonstrates training with mixed float16 datatypes for computations and float32 for variables. <br>

**Results (including charts/tables) and observations**  <br>
A graph to illustrate the architecture of our super model: <br>

<img width="747" alt="Screen Shot 2022-12-19 at 12 58 59 AM" src="https://user-images.githubusercontent.com/48727287/208357975-26d242e5-8ff6-48f7-b5af-e5dc662c887f.png">
<br>

Super Model - 60% Sparsity <br>
When testing polynomial decaying sparsity with a final sparsity of 60%, we find the supermodel outperforms the model with strictly pruning. Accuracy on the testing set is highest using the resnet 44; with both shallow and deep networks seeing a 1.5 times increase in testing accuracy.  When quantization is applied, the same networks are able to be stored In nearly half the bytes of the original model without dampening accuracy. This shows that the supermodel is superior in terms of size and accuracy. 
<br>
We achieved a Noticeable improvement in accuracy, for both Resnet20 and Resnet44.  <br>
The Best Overall model is ResNet 44 SuperModel. It achieved ~ x1.5 test accuracy improvement.  <br>
Size of quantized file is ~ x1.8 memory reduction for both.  <br>

<img width="302" alt="Screen Shot 2022-12-19 at 1 01 03 AM" src="https://user-images.githubusercontent.com/48727287/208358211-122fe9f2-0c2a-4475-ae5e-96f5a5623b5f.png">
<br>

Super Model - 70% Sparsity  <br>
Next, we attempted to fine-tune the supermodel slightly by keeping all the other parameters the same but changing final sparsity to 70%. The goal here would be to store an even smaller model, with 10% more zero weights, without losing much accuracy. The increase in test accuracy is similar to that of the model with 60% sparsity, losing just 1% accuracy between the best model for both, resnet 44. Although, there is little improvement in memory over the 60% sparsity model. When examined from the perspective of resnet 20, we do see a reduction of 1.8x the size, like with 60% sparsity.
<br>
We achieved a Noticeable improvement in accuracy, for both ResNet 20 and ResNet 44.   <br>
The Best Overall model is ResNet 44 SuperModel. It has ~ x1.5 test accuracy improvement.  <br>
The improvements of size of quantized file are listed below:   <br>
~ x1.8 memory reduction for Resnet 20  <br>
~ x1.4 memory reduction for Resnet 44  <br>

<img width="337" alt="Screen Shot 2022-12-19 at 1 02 16 AM" src="https://user-images.githubusercontent.com/48727287/208358370-88b4fea5-26ed-48cf-a28c-9aedbeb711a8.png">

<br>

The Best SuperModel  <br>
Overall, the model that performs best in terms of test accuracy is the resnet 44 supermodel with 60% sparsity. However, the loss in accuracy is very minimal between 60% and 70% sparsity. Given you are willing to sacrifice the one percent accuracy, you could opt for the other approach of using the 70% sparsity model. However, there are no noticeable decreases in the size of the quantized model that rounding doesn’t obscure. Perhaps there was a bug in the code, or somehow the quantization isn’t captured correctly for this model. However, we do see that quantization was properly applied for resnet 20, roughly halving memory. This makes sense because increasing the number of zeros should in theory improve quantization. Overall, if you were going to train this dataset with resnet 20, the supermodel with 70% sparsity would be the best choice for a very small model, faster training time and only 3% less accuracy.
<br>
The Best Accuracy is achieved by ResNet 44 SuperModel, 60% sparsity, Narrow margin with 70% sparsity.  <br>
The Best Size is ResNet 44 SuperModel, same for both sparsities.  <br>

<img width="380" alt="Screen Shot 2022-12-19 at 1 03 11 AM" src="https://user-images.githubusercontent.com/48727287/208358468-361a0170-5d90-4cf9-abd9-27cdd8af4344.png">

<img width="449" alt="Screen Shot 2022-12-19 at 1 03 22 AM" src="https://user-images.githubusercontent.com/48727287/208358484-d77f7191-9809-457f-ad7c-6be807c1c890.png">

Mixed Precision <br>
Overall, storing variables as float32 numbers while doing computations in float16 does speed up the training time for these two models, even on a GPU like a T4. This results in a speed up of 1.2 times for both networks. Also, average per epoch time is decreased as well. This method succeeds in decreasing training time without hindering accuracy, and there is even a 2% accuracy improvement for resnet20. Though these gains are small, they can't be discounted when deadlines are short and more models need to be trained. In order to isolate the effects of decreased training time to synchronous training, we chose not to include mixed precision in our supermodel. <br>

<img width="328" alt="Screen Shot 2022-12-19 at 1 04 21 AM" src="https://user-images.githubusercontent.com/48727287/208358639-93b99126-47e9-4992-9e0b-09fa1f36cc35.png">
<br>


