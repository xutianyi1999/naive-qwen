use burn::module::{Ignored, Module};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::{Backend, Tensor};
use burn::record::Recorder;
use burn::tensor::{activation, Int};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub tie_word_embeddings: bool,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}


impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device, cfg: &Config) -> Self {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        Mlp {
            gate_proj: LinearConfig::new(hidden_sz, intermediate_sz).with_bias(false).init(device),
            up_proj: LinearConfig::new(hidden_sz, intermediate_sz).with_bias(false).init(device),
            down_proj: LinearConfig::new(intermediate_sz, hidden_sz).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = activation::silu(gate);

        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.matmul(up))
    }
}

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,

    rotary_emb: Option<RotaryEncoding<B>>,

    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(device: &B::Device, cfg: &Config) -> Self {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        Attention {
            q_proj: LinearConfig::new(hidden_sz, num_heads * head_dim).with_bias(false).init(device),
            k_proj: LinearConfig::new(hidden_sz, num_kv_heads * head_dim).with_bias(false).init(device),
            v_proj: LinearConfig::new(hidden_sz, num_kv_heads * head_dim).with_bias(false).init(device),
            o_proj: LinearConfig::new(num_heads * head_dim, hidden_sz).with_bias(false).init(device),
            q_norm: RmsNormConfig::new(head_dim).with_epsilon(cfg.rms_norm_eps).init(device),
            k_norm: RmsNormConfig::new(head_dim).with_epsilon(cfg.rms_norm_eps).init(device),
            rotary_emb: Some(RotaryEncodingConfig::new(cfg.max_position_embeddings, head_dim).with_theta(cfg.rope_theta).init(device)),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        query
            .matmul(key.transpose())
            .div_scalar((self.head_dim as f32).sqrt())
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.dims();
        let b_sz = shape[0];
        let q_len = shape[1];
        let hidden_sz = shape[2];

        let q = self.q_proj.forward(x.clone())
            .reshape([b_sz, q_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let k = self.k_proj.forward(x.clone())
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let v = self.v_proj.forward(x.clone())
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let q = self.rotary_emb.as_ref().unwrap().forward(q);
        let k = self.rotary_emb.as_ref().unwrap().forward(k);

        let scroes = self.attn_scores(q, k);
        println!("scroes: {}", scroes);

        let t = activation::softmax(scroes, 3);
        println!("t: {}", t);

        let o = t.matmul(v)
            .swap_dims(1, 2)
            .reshape([b_sz, q_len, hidden_sz]);

        self.o_proj.forward(o)
    }
}

#[derive(Module, Debug)]
struct Layer<B: Backend> {
    attention: Attention<B>,
    mlp: Mlp<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Layer<B> {
    pub fn new(device: &B::Device, cfg: &Config) -> Self {
        Layer {
            attention: Attention::new(device, cfg),
            mlp: Mlp::new(device, cfg),
            input_layernorm: RmsNormConfig::new(cfg.hidden_size).with_epsilon(cfg.rms_norm_eps).init(device),
            post_attention_layernorm: RmsNormConfig::new(cfg.hidden_size).with_epsilon(cfg.rms_norm_eps).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();

        let x = self.input_layernorm.forward(x);
        let x = self.attention.forward(x);

        let x = x + residual;
        let residual = x.clone();

        let x = self.post_attention_layernorm.forward(x);
        let x= self.mlp.forward(x);

        residual + x
    }
}

#[derive(Module, Debug)]
pub struct Qwen3<B: Backend> {
    embedding: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3<B> {
    pub fn new(
        config: &Config,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);

        let layers = (0..config.num_hidden_layers)
            .map(|_| Layer::new(device, config))
            .collect();

        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        Qwen3 { embedding, layers, norm }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(x);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        self.norm.forward(x)
    }
}