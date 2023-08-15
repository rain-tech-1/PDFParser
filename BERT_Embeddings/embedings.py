import torch
from transformers import BertTokenizer, BertModel

tokenizer, model, quantized_model, device = None, None, None, None


def load_models(lang):
    global tokenizer, model, quantized_model, device

    model_name = "bert-base-en" if lang == "en" else "bert-base-chinese"
    # Load the tokenizer for traditional Chinese
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load the model for traditional Chinese
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cpu")
    # Ensure the model is using CPU
    model = model.to(device)
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8,  # the target dtype for quantized weights
    )
    quantized_model.eval()
    return model, quantized_model, tokenizer


def get_embeddings(text: str, lang: str = "en"):
    global tokenizer, quantized_model

    if not tokenizer:
        normal_model, quantized_model, tokenizer = load_models(lang)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Ensure no GPU is being used
    device = torch.device("cpu")
    inputs.to(device)

    # Quantised model
    outputs = quantized_model(**inputs)
    quantized_output = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    # normal model
    outputs = model(**inputs)
    output = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return quantized_output, output
