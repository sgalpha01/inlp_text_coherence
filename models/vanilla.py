import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BitsAndBytesConfig


class TransformerModel(nn.Module):
    def __init__(self, args, freeze_emb_layer=True):
        super(TransformerModel, self).__init__()
        # Using tf2 as variable name to maintain uniformity across the code
        config = None

        # Check if QLoRA is to be used and set up quantization config
        if args.use_qlora:
            args.logger.info("Configuring QLoRA with 4-bit precision...")
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # using nf4 for weights initialized from normal distribution
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # Load the model with or without quantization settings
        self.tf2 = AutoModel.from_pretrained(
            args.model_name,
            hidden_dropout_prob=args.dropout_rate,
            add_pooling_layer=False,
            config=config  # This will be None if not using QLoRA
        )

        # Prepare the model for training if QLoRA is used
        if args.use_qlora:
            self.tf2 = prepare_model_for_kbit_training(self.tf2)

        # Apply LoRA configurations if specified
        if args.use_lora or args.use_qlora:
            lora_alpha = args.lora_alpha
            lora_rank = args.lora_rank
            args.logger.info(f"Using LoRA with rank={lora_rank} and alpha={lora_alpha}")

            lora_config = LoraConfig(
                r=lora_rank,  # Rank for the low-rank approximation
                lora_alpha=lora_alpha,  # Scaling factor for LoRA
                target_modules=["query", "key", "value"],  # Typically apply LoRA to Q, K, and V of attention layers
                bias="none"  # No bias adjustments for simplicity
            )

            # Wrap the model with LoRA using PEFT
            self.tf2 = get_peft_model(self.tf2, lora_config)

        self.args = args
        if args.freeze_emb_layer and freeze_emb_layer:
            self.layer_freezing()

    def forward(self, input_data):
        input_ids, attention_mask = input_data
        output = self.tf2(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output[:, 0, :]  # Return the vector representation for [CLS] token from last layer output

    def layer_freezing(self, freeze_layers=[], freeze_embedding=True):
        if freeze_embedding:
            for param in self.tf2.embeddings.parameters():
                param.requires_grad = False
            self.args.logger.info("Froze embedding layer")

        for layer_idx in freeze_layers:
            for param in self.tf2.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
            self.args.logger.info(f"Froze internal layer: {layer_idx}")
