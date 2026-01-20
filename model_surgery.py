import torch


def model_surgery(my_model, hf_model, num_layers, vision_model=False):
    print("Starting transfer of weights from hf_model to my_model")

    if vision_model:
        my_module = my_model.vision_model
        hf_module = hf_model.vision_model
    else:
        my_module = my_model.text_model
        hf_module = hf_model.text_model

    # embeddings
    resp = my_module.embeddings.load_state_dict(hf_module.embeddings.state_dict())
    print(f"for {'vision' if vision_model else 'text'} embedding layers {resp}")

    # encoder layers
    for i in range(num_layers):
        hf_layer = hf_module.encoder.layers[i]
        my_layer = my_module.encoder.layers[i]

        # concatenate QKV weights
        qkv_weight = torch.cat(
            [
                hf_layer.self_attn.q_proj.weight,
                hf_layer.self_attn.k_proj.weight,
                hf_layer.self_attn.v_proj.weight,
            ],
            dim=0,
        )

        qkv_bias = torch.cat(
            [
                hf_layer.self_attn.q_proj.bias,
                hf_layer.self_attn.k_proj.bias,
                hf_layer.self_attn.v_proj.bias,
            ],
            dim=0,
        )

        state_dict = {
            "layer_norm1.weight": hf_layer.layer_norm1.weight,
            "layer_norm1.bias": hf_layer.layer_norm1.bias,
            "layer_norm2.weight": hf_layer.layer_norm2.weight,
            "layer_norm2.bias": hf_layer.layer_norm2.bias,
            "mlp.fc1.weight": hf_layer.mlp.fc1.weight,
            "mlp.fc1.bias": hf_layer.mlp.fc1.bias,
            "mlp.fc2.weight": hf_layer.mlp.fc2.weight,
            "mlp.fc2.bias": hf_layer.mlp.fc2.bias,
            "self_attn.qkv_proj.weight": qkv_weight,
            "self_attn.qkv_proj.bias": qkv_bias,
            "self_attn.out_proj.weight": hf_layer.self_attn.out_proj.weight,
            "self_attn.out_proj.bias": hf_layer.self_attn.out_proj.bias,
        }

        resp = my_layer.load_state_dict(state_dict)
        print(f"for layer {i} {resp}")

    # final norms / heads
    if vision_model:
        resp = my_module.post_layernorm.load_state_dict(
            hf_module.post_layernorm.state_dict()
        )
        print(f"for post layer norm {resp}")
    else:
        resp = my_module.final_layer_norm.load_state_dict(
            hf_module.final_layer_norm.state_dict()
        )
        print(f"for final norm layer {resp}")

    resp = my_module.head.load_state_dict(hf_module.head.state_dict())
    print(f"for final head {resp}")


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
