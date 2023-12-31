# -*- coding: utf-8 -*-
"""Copy of Copy of LLAMA-2-Fine-Tuning-Lora.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18TP41zHSodYJyZlArxS1H6kIR-3Mz7fZ
"""

#run this
from google.colab import drive
drive.mount('/content/drive')

#run this
!pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

#run this
train_dataset=load_dataset("csv",data_files="/content/drive/MyDrive/cleaned_dataset.csv")

#run this
!pip install langchain

#run this
from IPython.display import Markdown, display
from langchain.prompts import PromptTemplate
from peft import LoraConfig, get_peft_model

#run this
template = """Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer should be concise and encouraging.
Question: {Question}\n


### Answer: {Answer}"""

prompt = PromptTemplate(template=template, input_variables=["Question", 'Answer'])

#run this
# display sample to see template
sample = train_dataset['train'][0]
display(Markdown(prompt.format(Question=sample['Question'],

                               Answer=sample['Answer'])))

#run this
# Dataset


# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
refined_model = "llama-2-7b-mlabonne-enhanced"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# run this
def format_text(example):
    """ fill inputs in promt for a sample  """
    text = prompt.format(Question=example['Question'],

                         Answer=example['Answer'])
    return {"text": text}
train_dataset = train_dataset.map(format_text)

#dont run this
#LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=200,
    learning_rate=2e-3,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset['train'],
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
#fine_tuning.model.save_pretrained('/content/drive/MyDrive/llama2_finetunied')

# dont run this cell
# Save Model
fine_tuning.model.save_pretrained('/content/drive/MyDrive/trained-model-llama-7-b')

# run this
lora_config = LoraConfig.from_pretrained('/content/drive/MyDrive/trained-model-llama-7-b')
model = get_peft_model(base_model, lora_config)

##dont run this
from transformers import pipeline

# Load your model and tokenizer (assuming you've already loaded them)

while True:
    user_input = input("You: ")

    # Generate a response
    prompt = f"Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer should be concise and encouraging.\nQuestion: {user_input}\n\n"
    response = text_gen(prompt, max_length=200)[0]['generated_text']

    # Extract and display the response
    response = response.split('\n### Answer: ')[-1]
    print(f"Virtual Psych: {response}")

import re

#dont run this
from transformers import pipeline


text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=500)
# Define the introduction message outside the loop
introduction = " You are a chatbot named Virtual Psych built for mental health assistance. Answer the question considering context provided which includes conversation history,Your response should only include response and should not have question or context."

 # Flag to check if it's the first iteration
resp=[]
while True:
    user_input = input("You: ")
    x = " ".join(resp)

    # Generate a response
    prompt = f"{introduction}\nQuestion: {user_input}\nContext:{x}\n"
    response = text_gen(prompt)[0]['generated_text']
    pattern = r'(Virtual Psych:|Answer:|Response:)'  # Define the pattern to match either "Virtual Psych:" or "Answer:"
    split_text = re.split(pattern, response, 1)
    #print(response)  # Split the text based on the pattern
    if len(split_text) > 2:
        response = split_text[2].strip()
        resp.append(response)
        print(response)
    else:
        print(response)

# run this
from transformers import pipeline
import re

# Initialize the chatbot pipeline
text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=1024)

# Define the introduction message outside the loop
introduction = "Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer the question, it should be concise and encouraging."

# Initialize conversation history as a list with a maximum length
max_history_length = 5
conversation_history = []

while True:
    user_input = input("You: ")

    # Add user input to the conversation history
    conversation_history.append(f"You: {user_input}")

    # Trim the conversation history to the maximum length
    if len(conversation_history) > max_history_length:
        conversation_history.pop(0)  # Remove the oldest message

    # Generate a response
    context = "\n".join(conversation_history)
    prompt = f"{introduction}\nQuestion: {user_input}\nContext:\n{context}\n"
    response = text_gen(prompt)[0]['generated_text']
    pattern = r'(Virtual Psych:|Answer:)'  # Define the pattern to match either "Virtual Psych:" or "Answer:"
    split_text = re.split(pattern, response, 1)

    if len(split_text) > 2:
        response = split_text[2].strip()
        conversation_history.append(f"Virtual Psych: {response}")
        print(response)
    else:
        print("Bot response is incomplete or missing.")



#dont run this
from transformers import pipeline
import re

# Initialize the chatbot pipeline
text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=1024)

# Define the introduction message outside the loop
introduction = "Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer the question, it should be concise and encouraging."

# Initialize conversation history as a list with a maximum length
max_history_length = 5
conversation_history = []
bot_responses = []  # Store the chatbot's responses separately

while True:
    user_input = input("You: ")

    # Check if the user input is the same as the previous input
    if conversation_history and conversation_history[-1][3:] == user_input:
        print("Bot: I've already responded to that. Please ask something else.")
        continue

    # Add user input to the conversation history
    conversation_history.append(f"You: {user_input}")

    # Trim the conversation history to the maximum length
    if len(conversation_history) > max_history_length:
        conversation_history.pop(0)  # Remove the oldest message

    # Generate a response
    context = "\n".join(conversation_history)
    prompt = f"{introduction}\nQuestion: {user_input}\nContext:\n{context}\n"
    response = text_gen(prompt)[0]['generated_text']
    pattern = r'(Virtual Psych:|Answer:)'  # Define the pattern to match either "Virtual Psych:" or "Answer:"
    split_text = re.split(pattern, response, 1)

    if len(split_text) > 2:
        response = split_text[2].strip()
        conversation_history.append(f"Virtual Psych: {response}")
        bot_responses.append(response)  # Store the response separately
        print("Bot:", response)
    else:
        print("Bot response is incomplete or missing.")



templatew = """Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer should be concise and encouraging.
Question: {Question}\n


### Answer: {Answer}"""

prompt = PromptTemplate(template=templatew, input_variables=['Question','Answer'])

data = {
    'Question': "OK ALSO TELL ME HOW TO GET RID OF FINAL YEAR PROJECT OVERLOAD ?",
    'Answer': ""
}

prompt =(prompt.format(Question=data['Question'],

                               Answer=data['Answer']))
print(prompt)

# Generate Text

text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {prompt} [/INST]")
print(output[0]['generated_text'])

