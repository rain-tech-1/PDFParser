import torch
from transformers import BertTokenizer, BertModel

tokenizer, model, quantized_model, device = None, None, None, None


def load_models():
    global tokenizer, model, quantized_model, device
    # Load the tokenizer for traditional Chinese
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # Load the model for traditional Chinese
    model = BertModel.from_pretrained("bert-base-chinese")
    device = torch.device("cpu")
    # Ensure the model is using CPU
    model = model.to(device)
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8,  # the target dtype for quantized weights
    )
    quantized_model.eval()
    return quantized_model, tokenizer


def get_embeddings(text: str):
    global tokenizer, quantized_model

    if not tokenizer:
        quantized_model, tokenizer = load_models()

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Ensure no GPU is being used
    device = torch.device("cpu")
    inputs.to(device)

    outputs = quantized_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
