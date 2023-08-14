import torch
from transformers import BertTokenizer, BertModel

# Load the tokenizer for traditional Chinese
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


def get_embeddings(text: str):
    global tokenizer
    # Create a sample traditional Chinese text
    text = "我愛Python程式設計"

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Ensure no GPU is being used
    device = torch.device("cpu")
    inputs.to(device)

    # Load the model for traditional Chinese
    model = BertModel.from_pretrained("bert-base-chinese")

    # Ensure the model is using CPU
    model = model.to(device)

    # Run the data through the model
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
