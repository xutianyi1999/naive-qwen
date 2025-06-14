use std::path::PathBuf;
use burn::record::{HalfPrecisionSettings, Recorder};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use anyhow::Result;
use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::module::Module;
use burn::prelude::Shape;
use burn::tensor::{bf16, Distribution, Tensor};
use crate::qwen3::{Config, Qwen3};

mod qwen3;

fn launch() -> anyhow::Result<()> {
    let device = CudaDevice::default();

    let args = LoadArgs::new(PathBuf::from("D:/models/Qwen3-0.6B/model.safetensors"))
        // Example: Remove "model.encoder." prefix from keys
        .with_key_remap("\\bmodel\\.embed_tokens(\\b|\\.)", "embedding")
        .with_key_remap(r"^model\.norm\.(.+)", "norm.gamma")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.input_layernorm\.(.+)", "layers.$1.input_layernorm.gamma")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.(.+)", "layers.$1.post_attention_layernorm.gamma")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.(.+)", "layers.$1.mlp.gate_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.(.+)", "layers.$1.mlp.up_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.(.+)", "layers.$1.mlp.down_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.(.+)", "layers.$1.attention.q_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.(.+)", "layers.$1.attention.k_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.(.+)", "layers.$1.attention.v_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.(.+)", "layers.$1.attention.o_proj.$2")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.(.+)", "layers.$1.attention.q_norm.gamma")
        .with_key_remap(r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.(.+)", "layers.$1.attention.k_norm.gamma")

        .with_adapter_type(AdapterType::PyTorch); // Specify if adaptation is needed
        // .with_debug_print(); // Enable debug output

    let record = SafetensorsFileRecorder::<HalfPrecisionSettings>::default()
        .load(args, &device)?;

    let config: Config = serde_json::from_reader(std::fs::File::open("D:/models/Qwen3-0.6B/config.json")?)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    let qwen3 = Qwen3::<Cuda>::new(&config, &device).load_record(record);

    let tokens = Tensor::from_data([[1]], &device);
    qwen3.forward(tokens);
    Ok(())
}

fn main() {
    launch().unwrap();
}
