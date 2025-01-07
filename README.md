# Twitter Sentiment Analysis  

## Project Overview  
In the digital age, social media platforms like Twitter have become essential for understanding customer feedback. This project focuses on **Twitter Sentiment Analysis**, leveraging tweets from the airline industry to classify user sentiments into Positive, Negative and Neutral categories. By combining traditional machine learning models with deep learning techniques, the project seeks to identify patterns in user sentiment, enabling airlines to improve customer satisfaction and operational efficiency.  

This comprehensive analysis involves preprocessing raw text data, extracting features and training models to evaluate their performance. The project highlights a comparison between shallow machine learning models and deep learning approaches for sentiment classification.  

## Objectives  
The primary goals of this project include:  
1. **Sentiment Classification**: Categorize tweets into Positive, Negative, and Neutral sentiments.  
2. **Model Benchmarking**: Compare shallow and deep learning models to identify the most effective technique.  
3. **Textual Feature Engineering**: Employ advanced text preprocessing and embedding techniques to improve model performance.  
4. **Actionable Insights**: Generate insights from the analysis to support data-driven decisions in the airline industry.  

## Key Features  
- **Exploratory Data Analysis (EDA)**: Visualization and distribution analysis of sentiment classes across training, testing, and validation datasets.  
- **Text Preprocessing**: Cleaning and standardizing text data using techniques like tokenization, lemmatization, and stopword removal.  
- **Feature Extraction**: Utilize TF-IDF vectorization for traditional models and GloVe embeddings for deep learning.  
- **Model Performance Evaluation**: Evaluate models using metrics like the F1-score for a comprehensive comparison.  

## Tools & Technologies  
- **Python**: Programming language for end-to-end pipeline development.  
- **Pandas and NumPy**: Libraries for data manipulation and numerical operations.  
- **TF-IDF and GloVe Embeddings**: For text vectorization and word representation.  
- **Scikit-learn**: For training traditional machine learning models.  
- **Keras and TensorFlow**: Frameworks for implementing deep learning models.  
- **Matplotlib and Plotly**: For data visualization and interactive analysis.  

## Workflow Highlights  
1. **Data Collection**  
   - Extracted tweets related to the airline industry, along with sentiment labels.  
2. **Preprocessing Pipeline**  
   - Applied advanced text preprocessing methods, including tokenization, lemmatization, and stopword removal.  
   - Utilized GloVe embeddings for word representation in deep learning models.  
3. **Model Training**  
   - Trained and evaluated a combination of traditional and deep learning models.  
4. **Performance Comparison**  
   - Assessed models based on F1-scores to identify the most effective technique for sentiment classification.  

## Model Performance  

| **Model**                          | **F1-Score** |  
|-------------------------------------|--------------|  
| Naive Bayes                         | 0.7582       |  
| Random Forest                       | 0.7589       |  
| SVM Classifier                      | 0.7036       |  
| XGB Classifier                      | 0.7111       |  
| Convolutional Neural Network (CNN)  | 0.7418       |  
| LSTM Model                          | 0.7418       |  

The **Random Forest Classifier** achieved the highest F1-score among all models, demonstrating its ability to effectively capture sentiment patterns in the dataset.  

## Insights  
- **Class Imbalance**: Negative sentiments were the most prevalent across datasets, highlighting a trend of dissatisfaction among users.  
- **Preprocessing Impact**: Effective text preprocessing and embedding techniques significantly improved model accuracy.  
- **Model Comparison**: While traditional models like Random Forest performed exceptionally well, deep learning models (LSTM and CNN) provided comparable results, indicating their potential for further optimization.  

## Getting Started  

### Prerequisites  
- Python 3.8 or higher.  
- Install the required libraries.

## Results
The project demonstrates that incorporating contextual features into machine learning models leads to more accurate and reliable energy consumption predictions. The LSTM and GRU models performed significantly better than traditional models, with the Bidirectional LSTM providing the most accurate forecasts.
