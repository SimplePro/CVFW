# CVFW   (Computer Vision Feature Weight)
------------
##### definition
1. m is the horizontal size of the image, and n is the vertical size of the image.  
2. W is the weight matrix between pixels.  
3. P is one-dimensional data of image pixels.  


#### 1. Weights
![image](https://user-images.githubusercontent.com/66504341/129468258-8b12767e-d233-41c2-97d9-8293dbc12ef0.png)  
Weigihts formula  

![image](https://user-images.githubusercontent.com/66504341/129468268-d064d251-26f4-419a-aba7-3c8441176eb8.png)  
example  

#### 2. Weight grouping and feature grouping
- Grouping: As a preparatory step before finding a feature group, K pieces of W_ij collected from K pieces of training data are grouped. While contemplating the idea of ‘How to group?’, I decided to find the densely packed weights, and defined the group as follows. Group G_ij as shown in the figure.
(n_W_ij is the nth Wij.)  
![image](https://user-images.githubusercontent.com/66504341/129468346-e7de8ced-b65d-4ca6-b350-f51e497a2038.png)


- Feature group finding: There are variables that the user adjusts directly in the feature group finding, which can be seen as a practical learning step. FGN (Feature Group Number), which is the maximum number of feature groups allowed in one G_ij, and FWN (Feature Weight Number), which is the minimum number of elements in one feature group. The method to find the feature group is as follows. The feature group in G_ij is FW_ij (Feature Weight). In addition, FW_COUNT is used to define a cost function later.  
     
    Step 1. In one G_ij, the upper FGN groups are selected in the order of the largest number of elements.
    Step 2. Among the selected groups, the group that does not satisfy the minimum number of elements FWN is excluded.
    Step 3. The remaining groups after steps 1 and 2 are included in the feature group (FW_ij).
    Step 4. Converts the groups in the feature group to the mean.
    Step 5. If the number of elements in the feature group is more than 1, add 1 to FW_COUNT.
   
#### 3. Cost Function
- The cost function is the amount of difference between the trained image and the image to be compared. The cost function is an important function used when predicting a class. The factor entering the cost function is the weight of the image to be compared. FW_COUNT plays a role in adjusting the cost function well according to the feature group (FW) of the learned image. For example, if the number of FW_ij with the number of elements greater than 1 in the learned image A is three, the probability that Cost(W) is low is very high. (Because C(W_ij) = 0 if FW_ij is empty) Using FW_COUNT is for this case.

- A single cost function (cost function for one W_ij) is as follows. However, if FW_ij is empty, C(W_ij) = 0. (min is to find the minimum value among the matrices entered as arguments.)  
![image](https://user-images.githubusercontent.com/66504341/129468465-5866189e-d68e-4f88-ab48-04fe84dd573d.png)  


- The full cost function is  
![image](https://user-images.githubusercontent.com/66504341/129468472-82ea80e8-0105-4f85-83ba-76f1c1f47736.png)  


#### 4. Predict
- Image prediction is to predict the class of the image using the already learned image when there is an image to be predicted. Image prediction can be easily obtained using the cost function defined above.
![image](https://user-images.githubusercontent.com/66504341/129468513-6d6f75d6-4542-43b7-b59d-69d8edd7360e.png)  


#### 5. Feature Modeling
- Feature modeling is to synthesize the largest features in one learned class and return them as an image. For example, when learning a bag with different shapes, it finds and returns common features even in bags with different shapes. The modeled data is represented by M.
(W_ij is a feature group with the largest number of elements in FW_ij and N_ij is the maximum number of elements of groups in FW_ij)
![image](https://user-images.githubusercontent.com/66504341/129468551-88ffd141-e968-48b2-a45d-38470ae79c65.png)  

- Modeling Predict Class's Cost Function
![image](https://user-images.githubusercontent.com/66504341/129468564-a283b74a-99e4-46bf-9072-b1882bb6903d.png)  

