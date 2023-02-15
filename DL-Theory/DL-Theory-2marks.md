# Deep Learning - 2marks
## 15th Feb 2023

**1. What is ML?**

Machine learning is a subfield of artificial intelligence (AI) that involves the use of algorithms and statistical models to enable computer systems to learn from data, identify patterns, and make decisions or predictions without being explicitly programmed for each specific task.

In essence, machine learning enables computers to learn from experience, much like humans do. It involves training a machine learning model using a large dataset, which allows the system to recognize patterns and relationships between different types of data. Once the model is trained, it can be used to make predictions or decisions based on new data that it hasn't seen before.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model using labeled data, while unsupervised learning involves training a model on unlabeled data. Reinforcement learning involves training a model to make decisions based on feedback from its environment.

Machine learning is used in a wide range of applications, including image recognition, speech recognition, natural language processing, fraud detection, recommendation systems, and many others.

**1.1 Differentiate between data mining & ML** 
Data mining and machine learning are two related but distinct fields in the realm of artificial intelligence (AI). Although they share some similarities, there are significant differences between them.

Data mining is the process of discovering patterns and relationships in large datasets, often using statistical and computational techniques. It typically involves extracting information from data and using that information to make decisions. Data mining is often used in business and scientific research to identify patterns that can be used to make predictions or to understand complex relationships between different variables.

Machine learning, on the other hand, is a subfield of AI that involves building algorithms and models that enable machines to learn from data, make predictions, and perform tasks without being explicitly programmed. Machine learning algorithms are designed to automatically identify patterns and relationships in data and use that information to make decisions or predictions.

In other words, data mining is a way to extract insights and knowledge from large datasets, while machine learning is a way to build intelligent systems that can learn from data and improve their performance over time. Data mining is often used as a preprocessing step for machine learning, as it helps to identify important features and relationships in the data that can be used to train machine learning models.
Overall, while data mining and machine learning have some similarities, they are distinct fields that serve different purposes and use different techniques.

**2. What is overfitting in ML? Why overfitting happens.**
Overfitting is a common problem in machine learning where a model becomes too complex and fits the training data too well, leading to poor performance on new, unseen data. In other words, the model becomes too specialized in the training data and is not able to generalize well to new data.

Overfitting occurs when a machine learning model is trained on a limited or biased dataset, and it learns the noise or random fluctuations in the data as well as the underlying patterns. As a result, the model becomes too complex and captures the idiosyncrasies of the training data, rather than the underlying patterns that are generalizable to new data.

For example, if a model is trained on a dataset with a small number of samples, it may memorize the training data instead of learning the underlying patterns. This can lead to poor performance on new, unseen data. Similarly, if the model is too complex and has too many parameters, it may fit the training data too closely and fail to generalize well.

Overfitting can also occur if the model is trained on a biased dataset that does not represent the true distribution of the data. For example, if the training data is biased towards a particular class or feature, the model may learn to focus on that class or feature and perform poorly on new data that has a different distribution.

To prevent overfitting, machine learning practitioners use techniques like cross-validation, regularization, and early stopping. Cross-validation involves splitting the data into training and validation sets, allowing the model to be trained on the training set and evaluated on the validation set. Regularization involves adding a penalty term to the model's loss function to discourage overfitting. Early stopping involves stopping the training process when the validation error stops improving, preventing the model from overfitting the training data.

**3. Explain the functions of supervised and unsupervised learning.**
Supervised learning and unsupervised learning are two main categories of machine learning algorithms that are used to enable machines to learn from data.

Supervised learning is a type of machine learning in which the algorithm learns to map input data to output data based on a labeled dataset. In other words, the algorithm is given a dataset with both input and output data, and it learns to recognize patterns in the data and make predictions on new data based on these patterns. For example, a supervised learning algorithm can be trained on a dataset of images and their corresponding labels, and it can learn to recognize objects in new images based on the patterns it has learned from the training data.

Supervised learning algorithms can be further classified into regression and classification algorithms. Regression algorithms predict a continuous value, such as the price of a house, while classification algorithms predict a discrete value, such as whether an email is spam or not.

Unsupervised learning, on the other hand, is a type of machine learning in which the algorithm learns to find patterns and relationships in data without being given any labeled data. In other words, the algorithm is given a dataset without any specific labels, and it learns to group the data based on similarities or differences in the data. For example, an unsupervised learning algorithm can be used to group customers based on their purchasing behavior, without any specific knowledge about the types of products they buy.

Unsupervised learning algorithms can be further classified into clustering and association algorithms. Clustering algorithms group similar data points together, while association algorithms identify patterns and relationships between different data points.

The main function of supervised learning is to make predictions or classifications on new, unseen data based on the patterns learned from labeled training data. The main function of unsupervised learning is to discover hidden patterns and structures in data without any specific knowledge of the labels or outcomes. Both types of machine learning algorithms have their own unique applications and use cases, and they are often used in combination to solve complex problems.

**4. Explain the difference AI & ML**
AI (Artificial Intelligence) and ML (Machine Learning) are two related but distinct fields in the realm of computer science and engineering. While AI is a broad concept that refers to the creation of intelligent machines that can perform tasks that typically require human intelligence, ML is a specific subset of AI that focuses on building systems that can learn from data and improve their performance over time.

AI can be divided into two categories: narrow or weak AI, and general or strong AI. Narrow AI refers to AI systems that are designed to perform specific tasks, such as playing chess or recognizing speech, and can only operate within a narrow range of applications. General AI, on the other hand, refers to AI systems that are capable of performing a wide range of tasks and can think and reason like humans.

ML, on the other hand, is a subfield of AI that focuses on building algorithms and models that enable machines to learn from data, make predictions, and perform tasks without being explicitly programmed. ML algorithms are designed to automatically identify patterns and relationships in data and use that information to make decisions or predictions.

In other words, AI is a broad concept that encompasses the entire field of intelligent machines, while ML is a specific approach to building intelligent systems that can learn from data. ML is a key technology that enables many AI applications, such as natural language processing, image and speech recognition, and recommendation systems.

Overall, while AI and ML are related concepts, they are distinct fields with their own unique applications, methods, and challenges. AI is a broad and ambitious goal of creating machines that can think and reason like humans, while ML is a specific technique for building intelligent systems that can learn from data and improve their performance over time.

**5. What is cross validation, discuss the different types of cross validation**
Cross-validation is a technique used in machine learning to evaluate the performance of a model on unseen data. It involves dividing the data into several subsets, or "folds", and training and testing the model on different combinations of these folds. Cross-validation can help to prevent overfitting and provide a more accurate estimate of the model's performance on new, unseen data.

There are different types of cross-validation techniques, including:

***K-Fold Cross-Validation:*** This is one of the most commonly used cross-validation techniques. It involves dividing the data into k equal parts, or "folds". The model is then trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold used as the test set exactly once. The final performance of the model is then calculated as the average of the k test results.

***Stratified K-Fold Cross-Validation:*** This is similar to k-fold cross-validation, but it ensures that each fold contains a proportional representation of the classes in the data. This can be particularly useful in cases where the data is imbalanced, meaning that some classes have significantly fewer samples than others.

***Leave-One-Out Cross-Validation (LOOCV):*** This involves training the model on all data except for one sample, and testing the model on that sample. This process is repeated for each sample in the data, and the final performance of the model is calculated as the average of the results. LOOCV can be computationally expensive, particularly for large datasets.

***Repeated K-Fold Cross-Validation:*** This involves repeating the k-fold cross-validation process multiple times, with different random partitions of the data. This can help to improve the reliability of the model's performance estimate.

***Hold-Out Cross-Validation:*** This involves splitting the data into two parts, a training set and a test set. The model is trained on the training set and evaluated on the test set. This can be useful when the data is very large and k-fold cross-validation is computationally expensive, or when the goal is to compare the performance of different models on the same test set.

Overall, cross-validation is an important technique in machine learning for evaluating the performance of a model on unseen data and preventing overfitting. The choice of cross-validation technique will depend on the specific characteristics of the data and the problem at hand.

**6. Purpose of cross validation does cross validation reduce overfitting**
The main purpose of cross-validation in machine learning is to evaluate the performance of a model on unseen data and to prevent overfitting. Overfitting occurs when a model is too complex and has learned the noise or idiosyncrasies of the training data, resulting in poor generalization to new data.

Cross-validation can help to reduce overfitting by evaluating the performance of a model on multiple different subsets of the data, rather than just the single training set. By doing so, cross-validation provides a more accurate estimate of how the model will perform on new, unseen data. This helps to ensure that the model is not overfitting to the training data, but rather is learning more generalizable patterns in the data.

In addition to helping prevent overfitting, cross-validation can also help to:

Compare the performance of different models or model hyperparameters
Determine whether the model is underfitting, overfitting, or fitting the data well
Provide insight into the variability of the model's performance across different subsets of the data
Overall, cross-validation is a valuable technique in machine learning for evaluating model performance and reducing overfitting. It helps to ensure that the model is not simply memorizing the training data, but rather is learning more generalizable patterns that can be applied to new, unseen data.

**7. Compare AI,ML,NN**
AI (Artificial Intelligence), machine learning, and neural networks are related concepts but are not identical.

AI refers to the simulation of human intelligence in machines that are programmed to think and act like humans. It involves creating algorithms and intelligent systems that can perform tasks that typically require human intelligence, such as perception, reasoning, learning, and problem-solving.

Machine learning is a subset of AI that involves teaching machines to learn from data, rather than explicitly programming them. It is a type of statistical modeling that allows machines to automatically improve their performance on a task as they are exposed to more data. Machine learning algorithms can be divided into three main categories: supervised learning, unsupervised learning, and reinforcement learning.

Neural networks, also known as artificial neural networks (ANN), are a type of machine learning algorithm inspired by the structure and function of the human brain. They are composed of interconnected nodes (or "neurons") that process and transmit information. Neural networks are particularly well-suited for tasks such as image recognition and natural language processing, as they can learn to recognize complex patterns in data.

In summary, AI is a broad field that encompasses many different techniques and applications, including machine learning. Machine learning, in turn, is a subset of AI that involves teaching machines to learn from data, while neural networks are a specific type of machine learning algorithm that is inspired by the structure and function of the human brain.

**8. Two classification methods SVM can handle**
Support Vector Machines (SVM) is a powerful machine learning algorithm that can be used for both classification and regression tasks. When used for classification, SVM tries to find the hyperplane that best separates the classes in the feature space.

SVM can handle a wide range of classification tasks, including those where the classes are linearly separable or non-linearly separable. Some of the classification methods that SVM can handle include:

Binary Classification: In binary classification, the goal is to classify data into one of two classes. SVM is particularly well-suited for binary classification tasks, as it can find the optimal hyperplane that maximizes the margin between the two classes.

Multiclass Classification: In multiclass classification, the goal is to classify data into more than two classes. SVM can be used for multiclass classification by using one of two main approaches: One-vs-One (OVO) or One-vs-All (OVA). In the OVO approach, SVM trains one binary classifier for each pair of classes and makes a decision based on the majority vote. In the OVA approach, SVM trains one binary classifier for each class and makes a decision based on the classifier with the highest confidence score.

Overall, SVM is a versatile algorithm that can handle a wide range of classification tasks, making it a popular choice in machine learning.

**9. Differentiate between linearly separable data and non-linearly separable data.**

In machine learning, the term "linearly separable data" refers to data that can be separated into two or more classes by a straight line or a hyperplane. In other words, if the data can be separated by drawing a line or a plane, it is said to be linearly separable. For example, in a binary classification problem where the data consists of two features, if it is possible to draw a straight line that can completely separate the two classes, the data is said to be linearly separable.

On the other hand, "non-linearly separable data" refers to data that cannot be separated by a straight line or a hyperplane. In other words, if the data cannot be separated by a line or a plane, it is said to be non-linearly separable. For example, in a binary classification problem where the data consists of two features, if it is not possible to draw a straight line that can completely separate the two classes, the data is said to be non-linearly separable.

Non-linearly separable data can often be transformed into a higher-dimensional space where it becomes linearly separable. This is known as the "kernel trick," which involves mapping the original data to a higher-dimensional space where a linear classifier can be used. This allows non-linear decision boundaries to be learned in the higher-dimensional space, which can then be mapped back to the original feature space.

In summary, linearly separable data can be separated by a straight line or a hyperplane, while non-linearly separable data cannot be separated by a straight line or a hyperplane. Non-linearly separable data can often be transformed to a higher-dimensional space where a linear classifier can be used, using the kernel trick.
