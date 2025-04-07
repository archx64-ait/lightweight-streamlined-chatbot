# Lightweight streamlined Chatbot for IT Helpdesk Support

This group project for the AT82.05 Natural Language Understanding course is presented as a project proposal under the guidance of Dr. Chaklam Silpasuwanchai.

## Group Name: archx64

## Group Description

I'll know it when I see it.

## Team Members

- Kaung Sithu (<st124974@ait.asia>)
- Maung Maung Kyi Tha (<st125214@ait.asia>)
- Rida Fatma (<st125481@ait.asia>)

## Branches

This repository currently have 3 branches.

1. `main`: master branch of the repository.
2. `models`: this is for the models and jupyter notebooks
3. `dev`: development of the web applications

## Dataset and Processing

Ubuntu Dialog corpus will be used to train and evaluate our model. It contains roughly 1 million two-person conversations that provide real-world technical trouble shooting interactions. The conversations have an average of 8 turns each, with a minimum of 3 turns. All conversations are carried out in text form. The steps involved in preprocessing the data is descibed below:

1. A column named `folder` is removed as it's not relevant
2. Dialogue lengths are calculated by grouping messages based on `dialogueID`. The grouped dialogues are randomly shuffled to makeshare there is a representative sample.

    ```python
    dialogue_lengths = df.groupby('dialogueID').size().reset_index(name='num_rows')
    dialogue_lengths = dialogue_lengths.sample(frac=1, random_state=69)
    ```

3. Dialogues are selected sequentially until the cumulative number of rows reached the maximum threshold. Only 1000 rows are selected as a proof of concept and 100000 rows will be used for final training.

    ```python
    selected_ids = []
    total = 0
    max_rows = 1000

    for _, row in dialogue_lengths.iterrows():
        if total + row['num_rows'] > max_rows:
            break
        selected_ids.append(row['dialogueID'])
        total += row['num_rows']
    ```

4. The final dataset is created by filtering the original dataset to include only the selected dialogues. The resulting subset is saved to a CSV file for later use. This reduces the time and computational resources as it eliminates the need for rerunning the processing code before training the model.

    ```python
    subset_df = df[df['dialogueID'].isin(selected_ids)]
    subset_df.to_csv(f'sample/ubuntu_context_{max_rows}.csv')
    ```

## Training

The notebook file for training is `notebooks/training_gpt-neo-1.3B.ipynb`

### Dataset

We used the dataset created from the proprocessing step. The dialogues are converted into formatted prompt-response pairs:

```python
grouped = df.groupby("dialogueID")
conversations = []
for _, group in grouped:
    turns = group.sort_values("date")["text"].dropna().tolist()
    for i in range(len(turns) - 1):
        conversations.append({
            "text": f"### Prompt:\n{turns[i]}\n### Response:\n{turns[i + 1]}"
        })
```

After that, the dataset is splited into training 70%, validation 15% and test 15%.

### Tokenization

As most dialogues are short, prompts and reponses are tokenized with max length of 128 tokens, padding/truncating as necessary. Labels are prepared to match input IDs for causal language modeling.

### Model Configuration

GPT-Neo 1.3B is used as the base model. It is a transformer model designed using EleutherAI's replication of the GPT-3 architecture. GPT-Neo refers to the class of models, while 1.3B represents the number of parameters of this particular pre-trained model. It can be accessed from this link <https://huggingface.co/EleutherAI/gpt-neo-1.3B>

LoRA (Low-Rank Adaptation) significantly reduces memory consumption and computational overhead during fine-tuning. By updating only a few low-rank matrices within the attention layers instead of the entire model, LoRA achieves similar or better performance with far fewer parameters, enabling efficient training on limited hardware resources. `BitsAndBytes` is used to perform 4-bit quantization (nf4 type), significantly reducing the memory footprint and enabling the training of large models like GPT-Neo on hardware with limited GPU memory. It maintains model performance by carefully quantizing weights, thus balancing efficiency with accuracy. `TrainingArguments` from Hugging Face Transformers define the configuration and hyperparameters for training the model. Key parameters used include:

```python

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

training_args = TrainingArguments(
    output_dir="./models/gpt-neo",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    fp16=False,
    logging_dir="./logs",
    logging_steps=100,
    logging_strategy='steps',
    save_strategy="epoch",
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=100,
    report_to=None
)
```

Training and validaton losses are tracked and visualized after training to monitor performance and convergence. ![Description](figures/train_valid_loss.PNG)

### Results

After training, the final model and tokenizer checkpoints saved uploaded and Google Drive. It can be accessed from this link. <https://drive.google.com/drive/folders/1TuFHKKmRJYEOV_haWLPWsAvOxi32x9m9?usp=sharing>

## Web Application

The web applcation, Lightweight(LW) Streamlined Chatbot, leverages a GPT-Neo 1.3B model fine-tuned with Low-Rank Adaptation (LoRA) to provide accurate and context-aware responses. It has simple and intuitive user interface powered by Streamlit. Users can control the randomness of chatbot responses.

### Running the application

Ensure you have the required dependencies installed:

```bash
pip install streamlit transformers peft bitsandbytes torch
```

Execute the following command to run the application. The web app will open in your default web browser.

```bash
streamlit run app/streamlit_app.py
```

### How to use?

1. Enter your question or prompt into the provided input box.
2. Use the slider to adjust the creativity (temperature) of the responses. Lower values produce more predictable answers, while higher values provide more creative, varied outputs.
3. The chatbot maintains context by displaying previous interactions.

## Contribution

Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests.
