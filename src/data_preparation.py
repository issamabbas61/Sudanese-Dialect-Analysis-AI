# 2_Model_Training.ipynb

# Make sure you have run 1_Data_Preparation.ipynb or load the dataset
# If loading:
# from datasets import load_from_disk
# dataset_dict = load_from_disk("data/processed_dialect_dataset")

# For this example, let's assume dataset_dict is available from previous cell execution
# If running notebooks separately, copy the data loading part from 1_Data_Preparation.ipynb
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import json

file_path = '../data/dialect_samples1.json'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
df = pd.DataFrame(raw_data)
unique_labels = df['label'].unique().tolist()
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for i, label in enumerate(unique_labels)}
df['labels'] = df['label'].map(label_to_id)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
hf_train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
hf_test_dataset = Dataset.from_pandas(test_df).remove_columns(["__index_level_0__"])
dataset_dict = DatasetDict({'train': hf_train_dataset, 'test': hf_test_dataset})

#from datasets import load_from_disk
#dataset_dict = load_from_disk("../data/processed_dialect_dataset")



# --- Tokenization ---
from transformers import AutoTokenizer

MODEL_NAME = "aubmindlab/bert-base-arabertv02" # Or "CAMeL-Lab/bert-base-arabic-camelbert-mix"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # 'text' is the column name containing the proverb/sentence
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# Remove original columns and set format to PyTorch tensors
tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])
tokenized_datasets.set_format("torch")

print("--- Tokenized Dataset Sample ---")
print(tokenized_datasets['train'][0])
print("-" * 20)

# Continue in 2_Model_Training.ipynb

from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np

# Load the pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
MODEL_NAME,
num_labels=len(unique_labels),
id2label=id_to_label,
label2id=label_to_id
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on: {device}")

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
# Use 'weighted' average for f1, precision, recall in multi-class if imbalanced
    precision, recall, f1, _ = precision_recall_fscore_support(
    labels, predictions, average='weighted', zero_division=0
)

    return {
    'accuracy': acc,
    'f1': f1,
    'precision': precision,
    'recall': recall
}

# Continue in 2_Model_Training.ipynb

from transformers import TrainingArguments, Trainer

    # Define training arguments
training_args = TrainingArguments( 
    output_dir="../models/dialect_id_model1",       # Where to save model checkpoints
    eval_strategy="epoch",           # Evaluate after each epoch
    learning_rate=2e-5,                    # Standard fine-tuning LR
    per_device_train_batch_size=8,         # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=5,                    # Number of training epochs
    weight_decay=0.01,                     # L2 regularization
    logging_dir='../logs',                  # For TensorBoard logs
    logging_steps=50,                      # Log every 50 steps
    save_strategy="epoch",                 # Save model after each epoch
    load_best_model_at_end=True,           # Load best model at end of training
    metric_for_best_model="f1",            # Monitor F1-score for best model
    push_to_hub=False,                     # Change to True if you want to push to HF Hub
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer # Pass tokenizer for proper handling
)

# Start training!
print("--- Starting Model Fine-tuning ---")
trainer.train()

# Evaluate on the test set after training
print("\n--- Final Evaluation on Test Set ---")
results = trainer.evaluate()
print(results)

# Save the fine-tuned model and tokenizer
model.save_pretrained("../models/fine_tuned_dialect_id_model1")
tokenizer.save_pretrained("../models/fine_tuned_dialect_id_model1")
print("\nModel saved to ../models/fine_tuned_dialect_id_model1")