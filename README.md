# Fall Detection - Project

## Target

Find the best classification algorithm between:

 - Support Vector Machine 
 - Random Forest Tree
 - K-nearest Neighbor
 - Baesyan decision making (?)

## Dataset
The dataset is available on Kaggle (at: [Fall Detection - Dataset](https://www.kaggle.com/pitasr/falldata/version/1)).
### Desctription
Falls among the elderly is an important health issue. Fall detection and movement tracking are therefore instrumental in addressing this issue. This paper responds to the challenge of classifying different movements as a part of a system designed to fulfill the need for a wearable device to collect data for fall and near-fall analysis. Four different fall trajectories (forward, backward, left and right), three normal activities (standing, walking and lying down) and near-fall situations are identified and detected.

Falls are a serious public health problem and possibly life threatening for people in fall risk groups. We develop an automated fall detection system with wearable motion sensor units fitted to the subjects’ body at six different positions. Each unit comprises three tri-axial devices (accelerometer, gyroscope, and magnetometer/compass). Fourteen volunteers perform a standardized set of movements including 20 voluntary falls and 16 activities of daily living (ADLs), resulting in a large dataset with 2520 trials. To reduce the computational complexity of training and testing the classifiers, we focus on the raw data for each sensor in a 4 s time window around the point of peak total acceleration of the waist sensor, and then perform feature extraction and reduction.

We successfully distinguish falls from ADLs using six machine learning techniques (classifiers): the k-nearest neighbor (k-NN) classifier, least squares method (LSM), support vector machines (SVM), Bayesian decision making (BDM), dynamic time warping (DTW), and artificial neural networks (ANNs). We compare the performance and the computational complexity of the classifiers and achieve the best results with the k-NN classifier and LSM, with sensitivity, specificity, and accuracy all above 95%. These classifiers also have acceptable computational requirements for training and testing. Our approach would be applicable in real-world scenarios where data records of indeterminate length, containing multiple activities in sequence, are recorded.

_Özdemir, Ahmet Turan, and Billur Barshan. “Detecting Falls with Wearable Sensors Using Machine Learning Techniques.” Sensors (Basel, Switzerland) 14.6 (2014): 10691–10708. PMC. Web. 23 Apr. 2017._

## Tech Spec
**The Dataset is composed of 6 features**:

 1. TIME: Monitoring time
 2. SL: Sugar level
 3. EEG: EEG monitoring rate
 4. BP: Blood pressure
 5. HR: Heart beat rate
 6. CIRCULATION: Blood circulation

**And 6 Classes (value associated between brackets)**:

- Standing (0)
- Walking (1)
- Sitting (2)
- Falling (3)
- Cramps (4)
- Running (5)

As shown, the number of classes is high and is not good for classification problem. The comparison of these **three or four** algorithm helps to detect which ones is the best on this kind of problems