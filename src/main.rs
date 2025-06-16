use std::io::Write;
use crate::qwen3::{Config, Qwen3};
use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::backend::Cuda;
use burn::module::Module;
use burn::prelude::Shape;
use burn::record::{HalfPrecisionSettings, Record, Recorder};
use burn::tensor::{bf16, DType, Distribution, Tensor, TensorData};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use std::path::{Path, PathBuf};

mod qwen3;

fn launch(base_path: &Path) -> anyhow::Result<()> {
    let device = CudaDevice::default();

    let args = LoadArgs::new(PathBuf::from(base_path.join("model.safetensors")))
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
        .with_adapter_type(AdapterType::PyTorch);

    let record = SafetensorsFileRecorder::<HalfPrecisionSettings>::default().load(args, &device)?;

    let config: Config =
        serde_json::from_reader(std::fs::File::open(base_path.join("config.json"))?)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    let qwen3 = Qwen3::<Cuda>::new(&config, &device).load_record(record);
    let tokenizer = tokenizers::Tokenizer::from_file(base_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokens: {}", e))?;

    let prompt = "Hello, my name is";

    let tokens = tokenizer.encode_fast(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to load tokens: {}", e))?;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    let mut tokens = tokens.get_ids().to_vec();
    let mut decode_stream = tokenizer.decode_stream(false);

    loop {
        let data = TensorData::new(tokens.clone(), [1, tokens.len()]);
        let input = Tensor::from_data(data, &device);

        let logits = qwen3.forward(input);

        let dims = logits.dims();
        let logits = logits.slice([0..dims[0], dims[1] - 1..dims[1]]);
        let last_logits = logits.argmax(2);

        let data = last_logits.into_data();
        assert_eq!(data.dtype, DType::I32);
        let out_tokens: Vec<i32> = data.into_vec().map_err(|e| anyhow::anyhow!("mismatch dtype"))?;

        assert_eq!(out_tokens.len(), 1);

        if let Some(s) = decode_stream.step(out_tokens[0] as u32).unwrap() {
            print!("{}", s);
            std::io::stdout().flush()?;
        }

        tokens.push(out_tokens[0] as u32);
    }

    Ok(())
}

fn main() {
    let args = std::env::args();
    let base_path = args.skip(1).next().expect("expect base path argument");

    let base_path = Path::new(&base_path);
    launch(base_path).unwrap();
}
