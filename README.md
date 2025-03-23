# **Personalized AI Chatbot - Fine-Tuning Mistral 7B on WhatsApp Chats**

This project fine-tunes **Mistral 7B** on personal WhatsApp chats using **QLoRA** for memory-efficient training. The goal is to create an AI chatbot that mirrors your personal texting style.

---

## **📌 Features**
- Fine-tunes **Mistral 7B** on personal chat data.
- Uses **QLoRA** for efficient fine-tuning on consumer GPUs.
- Deployable via **Hugging Face** or locally.

---

## **🛠️ Setup Instructions**

### **1️⃣ Load Model from Hugging Face**
First, authenticate and load the **Mistral 7B** base model from Hugging Face.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Authenticate Hugging Face (optional for public models)
from huggingface_hub import login
login("your_huggingface_token")

# Load Model & Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("✅ Mistral 7B Model Loaded Successfully!")
```

---

### **2️⃣ Export WhatsApp Chats**
1. Open **WhatsApp** → Select a chat → Tap `⋮ More Options` → `Export Chat`.
2. Choose `Without Media` and save it as a `.txt` file.
3. Repeat for multiple chats and store them in a ZIP folder.

---

### **3️⃣ Convert Chat Data to JSONL Format**

Convert exported chats into a structured dataset for fine-tuning.

```python
import json
import re

def process_chat(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    message_pattern = re.compile(r"^(.*?):\s(.*)")

    for line in lines:
        match = message_pattern.match(line.strip())
        if match:
            sender, message = match.groups()
            if data:
                data.append({"instruction": data[-1]["response"], "response": message})
            else:
                data.append({"instruction": "User starts conversation", "response": message})
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

process_chat("whatsapp_chat.txt", "formatted_chats.jsonl")
print("✅ Chat Data Processed Successfully!")
```

---

### **4️⃣ Upload Dataset to Hugging Face (Optional)**

```python
from datasets import load_dataset, DatasetDict

# Load dataset to Hugging Face Hub
dataset = load_dataset("json", data_files="formatted_chats.jsonl")
dataset.push_to_hub("your_huggingface_username/your_dataset_name")
```

---

### **5️⃣ Fine-Tune Mistral 7B Using QLoRA**

```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# LoRA Configuration
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# Train-Test Split
dataset = dataset["train"].train_test_split(test_size=0.1)

# Fine-Tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args
)
trainer.train()
```

---

### **6️⃣ Merge Fine-Tuned Weights & Save Model**

```python
from peft import PeftModel

# Merge LoRA Weights
merged_model = model.merge_and_unload()
merged_model.save_pretrained("mistral-7b-merged")
tokenizer.save_pretrained("mistral-7b-merged")

print("✅ Fine-Tuned Model Merged Successfully!")
```

---

### **7️⃣ Upload Fine-Tuned Model to Hugging Face**

```python
from huggingface_hub import upload_folder
upload_folder(repo_id="your_huggingface_username/mistral-finetuned", folder_path="mistral-7b-merged")
print("✅ Fine-Tuned Model Uploaded!")
```

---

### **8️⃣ Test Fine-Tuned Model**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Fine-Tuned Model
model_name = "your_huggingface_username/mistral-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Generate Response
def generate_response(prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test Chatbot
response = generate_response("Hey, how are you?")
print("\nFine-Tuned Model Response:\n", response)
```

---

## **🚀 Future Improvements**
- Implement **Retrieval-Augmented Generation (RAG)** for factual accuracy.
- Deploy chatbot via **WhatsApp Web Bot**.
- Optimize response generation speed.

## **📌 Conclusion**
This project successfully fine-tunes **Mistral 7B** on WhatsApp chat data using **QLoRA**, enabling the chatbot to generate responses in a **personalized texting style**.

---

🔹 **Author:** [Zain Yasir](https://github.com/ZainYasir)  
🔹 **License:** MIT  
🔹 **Contributions & Issues:** Feel free to open a PR or issue! 🚀

