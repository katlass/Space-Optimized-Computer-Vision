# Space Optimized CNN with Synchronous Distributed Training, Weight Pruning, and Quantization in Vertex AI on GCP with TensorFlow Lite

**A description of the project**  <br> 
Developed several space optimized CNNs for image classification through the strategies listed above while trying more shallow resnet20 and deeper resnet44 with various sparsity decaying strategies and data augmentation. The best model is then selected for use.


**A description of the repository** <br>
This repository contains the codes and all relevant training logs/models produced during the project. <br>
The training logs and models for ResNet 20 and ResNet 44 are stored in separate folders respectively.<br>

Under the folder /codes included is all training codes for the trials of exploration. These include testing constant vs. polynomially decaying sparsity in weight pruning and also mixed precision vs. full precision training on both CIFAR-10 and CIFAR-100 datasets. <br>

The final results are two different models, one with final sparsity 60% and one with 70% sparsity. The notebooks that produced the results of the final model trials are sixty_sparsity.ipynb and seventy_sparsity.ipynb in the /codes folder. <br>

Hyperparameters are found through 29 trial runs on a purely pruning based model. Maximal sparsity 60% was selected; it gives the desired validation accuracy. The training log and trained models are stored under /Final_Model/super_0.6 and /Final_Model/super_0.7 <br>

**Example commands to execute the code**   <br>     
I've included all codes as jupyter notebooks, and all the notebooks can be run on GCP VMs. Note that for the distributed training of the model, it needs to be run on Vertex AI. This will vastly speed up training time. The set up of Vertex AI is described in detail in this video: https://www.youtube.com/watch?v=rAGauhXYgw4&list=WL&index=1 . When logged into Vertex AI workbench, press the "JupyterLab" button to launch jupyterlab, and upload the jupyter notebook using the UI. Then, on the upper-right corner, select the machine configuration. 2 Tesla V100 GPUs each with 4 CPUs and 15 GB RAM were used. Now, the distributed training notebook can be run just as a standard jupyter notebook. Run final_demo.ipynb to train a Cifar100 model. You must first change the logname in resnet_training() to your appropriate file path to store training data. Feel free to experiment with hyperparameters such as initial/final sparsity, pruning frequency, pruning schedule, etc as instructed in the demo notebook. Another notebook, mixed_precision.ipynb is also included which demonstrates training with mixed float16 datatypes for computations and float32 for variables. <br>

**Results (including charts/tables) and observations**  <br>
<img width="747" alt="Screen Shot 2022-12-19 at 12 58 59 AM" src="https://user-images.githubusercontent.com/48727287/208357975-26d242e5-8ff6-48f7-b5af-e5dc662c887f.png">
<br>

**60% Sparsity Model** <br>
When testing polynomial decaying sparsity with a final sparsity of 60%, introducing data augmentation more than offsets the lost accuracy from fewer neurons. Accuracy on the testing set is highest using the resnet 44, with both shallow and deep networks seeing a 1.5x increase. When quantization is applied using TensorFlow Lite, the same networks can be stored in nearly half the bytes of the original model without dampening accuracy. This model is thus superior in terms of size and accuracy. 
<br>
The Best Overall model is ResNet 44 Model: <br>
~ It achieved ~ x1.5 test accuracy improvement.  <br>
~ Size of quantized file is ~ x1.8 memory reduction for both.  <br>

<img width="302" alt="Screen Shot 2022-12-19 at 1 01 03 AM" src="https://user-images.githubusercontent.com/48727287/208358211-122fe9f2-0c2a-4475-ae5e-96f5a5623b5f.png">
<br>

**70% Sparsity Model**  <br>
Next, the model was fine-tuned slightly by keeping all the other parameters the same but changing final sparsity to 70%. The goal here would be to store an even smaller model, with 10% more zero weights, without losing much accuracy. The increase in test accuracy is similar to that of the model with 60% sparsity, losing just 1% accuracy for this size reduction. 
<br>
The Best Overall model is ResNet 44 Model. It has ~ x1.5 test accuracy improvement.  <br>
The improvements of size of quantized file are listed below:   <br>
~ x1.8 memory reduction for Resnet 20  <br>
~ x1.4 memory reduction for Resnet 44  <br>

<img width="337" alt="Screen Shot 2022-12-19 at 1 02 16 AM" src="https://user-images.githubusercontent.com/48727287/208358370-88b4fea5-26ed-48cf-a28c-9aedbeb711a8.png">

<br>

**The Best Model**  <br>
Overall, the model that performs best in terms of test accuracy is the resnet 44 model with data augmentation and 60% polynomially decaying sparsity. However, the loss in accuracy is very minimal between 60% and 70% sparsity. Given you are willing to sacrifice the one percent accuracy, you could opt for the other approach of using the 70% sparsity model. Quantization roughly halves memory, as observed in these models. Overall, if you were going to train this dataset with resnet 20, the optimized model with 70% sparsity would be the best choice for a very small model, faster training time and only 3% less accuracy.
<br>
The best accuracy is achieved by the ResNet 44 model with 60% sparsity, narrow margin with 70% sparsity.  <br>

<img width="380" alt="Screen Shot 2022-12-19 at 1 03 11 AM" src="https://user-images.githubusercontent.com/48727287/208358468-361a0170-5d90-4cf9-abd9-27cdd8af4344.png">

<img width="449" alt="Screen Shot 2022-12-19 at 1 03 22 AM" src="https://user-images.githubusercontent.com/48727287/208358484-d77f7191-9809-457f-ad7c-6be807c1c890.png">

**Mixed Precision** <br>
Overall, storing variables as float32 numbers while doing computations in float16 does speed up the training time for these two models, even on a GPU like a T4. This results in a speed up of 1.2 times for both networks. Also, average per epoch time is decreased as well. This method succeeds in decreasing training time without hindering accuracy, and there is even a 2% accuracy improvement for resnet20. Though these gains are small, they can't be discounted when deadlines are short and more models need to be trained. In order to isolate the effects of decreased training time to synchronous training, mixed precision was not included in the final model. <br>

<img width="328" alt="Screen Shot 2022-12-19 at 1 04 21 AM" src="https://user-images.githubusercontent.com/48727287/208358639-93b99126-47e9-4992-9e0b-09fa1f36cc35.png">
<br>


