# Web History Analysis

**Web History Analysis** is an advanced tool for classifying and categorizing URLs from browser history logs using machine learning techniques. This project leverages deep learning models, specifically an LSTM (Long Short-Term Memory) network, to classify URLs into predefined categories based on historical browsing data. It’s ideal for security researchers, data analysts, or anyone interested in analyzing web browsing activity and categorizing web traffic effectively.

![screenshot](https://github.com/yogsec/Web-History-Analysis/blob/main/Hack-with-ai.jpeg)

Designed by **YogSec**, Web History Analysis is a powerful solution to analyze large sets of URLs, offering valuable insights into user browsing patterns and website classifications.

## Key Features

- **URL Classification**: Automatically classifies URLs from browser history logs into predefined categories.
- **Machine Learning Integration**: Utilizes TensorFlow's LSTM network to classify URLs based on labeled training data.
- **Preprocessing Capabilities**: Cleans and processes URLs to remove unwanted parts such as protocols, numbers, and special characters.
- **File Input Support**: Classify a list of URLs from a CSV file or text file, making it easy to work with large datasets.
- **Model Evaluation**: After training, the model evaluates its performance using a test dataset, providing an accuracy report.

## Installation

Before using Web History Analysis, make sure to install the necessary dependencies. The tool requires Python 3.x and the following Python packages:

- **TensorFlow**: A deep learning library to train the model.
- **pandas**: For handling and processing CSV data.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning utilities such as label encoding and train-test splitting.

### Install the required dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn
```

## How to Use

Follow these simple steps to get started with **Web History Analysis**:

### 1. Prepare Your Labeled Data

You need to prepare a CSV file (`labeled_data.csv`) with the following structure:

- `url`: The URL from your browser history.
- `category`: The category that the URL belongs to (e.g., Shopping, News, Social Media, etc.).

Example (`labeled_data.csv`):

| url                                       | category     |
|-------------------------------------------|--------------|
| https://www.example.com                   | Shopping     |
| https://news.example.com                  | News         |
| https://www.facebook.com                  | Social Media |

### 2. Preprocessing and Model Training

The code will load the labeled data and preprocess the URLs by removing the protocol (http, https), replacing numbers with a placeholder, and cleaning up special characters. Then, it tokenizes and pads the URLs to make them compatible with the deep learning model. After this, the LSTM model is trained on the preprocessed data.

### Example of how the training works:

```python
import pandas as pd

df = pd.read_csv('labeled_data.csv')  # Load your labeled data
```

Once the data is prepared, the training process starts and will automatically evaluate the model's accuracy on a test dataset.

### 3. Classify URLs

Once the model is trained, you can classify URLs from any file (e.g., CSV or text). This is done using the `classify_urls_from_file()` function. It processes the URLs, applies the model for classification, and outputs the predicted categories.

To classify URLs from a file:

```bash
python web_history_analysis.py
Enter the filename containing URLs: urls.txt
```

### Example of the output:

```
URL: https://example.com/product/123 → Category: Shopping
URL: https://news.example.com/article/456 → Category: News
```

### 4. Evaluate Model Accuracy

After training, the model will evaluate its accuracy on the test set and output the result.

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

## File Structure

Here’s an example structure for the project:

```
.
├── labeled_data.csv        # CSV file with labeled URLs and categories
├── web_history_analysis.py # The script to train the model and classify URLs
├── urls.txt                # A text file containing URLs to be classified
└── README.md               # This README file
```

## Licensing

This project is licensed under the **MIT License**. Feel free to fork, modify, and distribute this tool as per your needs.

## 🌟 Let's Connect!

Hello, Hacker! 👋 We'd love to stay connected with you. Reach out to us on any of these platforms and let's build something amazing together:

🌐 **Website:** [https://yogsec.github.io/yogsec/](https://yogsec.github.io/yogsec/)  
📜 **Linktree:** [https://linktr.ee/yogsec](https://linktr.ee/yogsec)  
🔗 **GitHub:** [https://github.com/yogsec](https://github.com/yogsec)  
💼 **LinkedIn (Company):** [https://www.linkedin.com/company/yogsec/](https://www.linkedin.com/company/yogsec/)  
📷 **Instagram:** [https://www.instagram.com/yogsec.io/](https://www.instagram.com/yogsec.io/)  
🐦 **Twitter (X):** [https://x.com/yogsec](https://x.com/yogsec)  
👨‍💼 **Personal LinkedIn:** [https://www.linkedin.com/in/bug-bounty-hunter/](https://www.linkedin.com/in/bug-bounty-hunter/)  
📧 **Email:** abhinavsingwal@gmail.com

---

## ☕ Buy Me a Coffee

If you find our work helpful and would like to support us, consider buying us a coffee. Your support keeps us motivated and helps us create more awesome content. ❤️

☕ **Support Us Here:** [https://buymeacoffee.com/yogsec](https://buymeacoffee.com/yogsec)

Thank you for your support! 🚀

**Designed by YogSec** - A cybersecurity startup focused on vulnerability assessment and security research.

For any questions or feedback, contact us via email at [abhinavsingwal@gmail.com](mailto:abhinavsingwal@gmail.com) or visit our [LinkedIn](https://www.linkedin.com/in/bug-bounty-hunter).
