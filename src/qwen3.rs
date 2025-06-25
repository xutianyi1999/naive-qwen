use burn::module::{Module, Param, ParamId};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::{Backend, Device, Tensor};
use burn::record::Recorder;
use burn::tensor::{activation, Int, TensorData};
use serde::Deserialize;
use std::ops::Deref;

pub struct KVCache<B: Backend> {
    // (batch_size, num_heads, seq_len, head_dim)
    keys: Option<Tensor<B, 4>>,
    values: Option<Tensor<B, 4>>,
}

impl <B: Backend> KVCache<B> {
    pub fn new() -> Self {
        KVCache {
            keys: None,
            values: None,
        }
    }

    fn put(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) {
        match (self.keys.clone(), self.values.clone()) {
            (None, None) => {
                self.keys = Some(k);
                self.values = Some(v);
            }
            (Some(keys), Some(values)) => {
                self.keys = Some(Tensor::cat(vec![keys, k], 2));
                self.values = Some(Tensor::cat(vec![values, v], 2));
            }
            _ => panic!("keys and values must be coercible")
        }
    }

    fn offset(&self) -> usize {
        self.keys.as_ref().map(|keys| keys.dims()[2]).unwrap_or(0)
    }
}

fn rotate_half<B: Backend>(xs: Tensor<B, 4>) -> Tensor<B, 4> {
    let last_dim = xs.dims()[3];
    let xs1 = xs.clone().narrow(3, 0, last_dim / 2);
    let xs2 = xs.narrow(3, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat(vec![xs2.neg(), xs1], 3)
}

pub fn rope_slow<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
    let [_b_sz, _h, seq_len, _n_embd] = x.dims();
    let cos = Tensor::cat(vec![cos.clone(), cos], 1);
    let sin = Tensor::cat(vec![sin.clone(), sin], 1);
    let cos = cos.narrow(0, 0, seq_len);
    let sin = sin.narrow(0, 0, seq_len);
    let cos = cos.unsqueeze();
    let sin = sin.unsqueeze();
    x.clone().mul(cos) + rotate_half(x).mul(sin)
}

#[derive(Module, Debug)]
struct RotaryEmbedding<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>
}

impl <B: Backend> RotaryEmbedding<B> {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device<B>,
    ) -> Self {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq= Tensor::from_data(TensorData::new(inv_freq, [1, inv_freq_len]), device);
        let t: Tensor<B, 2> = Tensor::arange(0i64..max_position_embeddings as i64, device)
            .float()
            .reshape([max_position_embeddings, 1]);

        let freqs = t.matmul(inv_freq);
        let sin = freqs.clone().sin();
        let cos = freqs.cos();

        Self {
            sin,
            cos
        }
    }

    fn forward(&self, x: Tensor<B, 4>, offset: usize) -> Tensor<B, 4> {
        let [_b_sz, _qh, seq_len, _n_embd] = x.dims();
        let cos = self.cos.clone().narrow(0, offset, seq_len);
        let sin = self.sin.clone().narrow(0, offset, seq_len);

        rope_slow(x, cos, sin)
    }
}

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
        let t = gate * up;
        let out = self.down_proj.forward(t);
        out
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

    rotary_emb: Option<RotaryEmbedding<B>>,

    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

pub fn repeat_interleave<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    let [a, b, c, d] = x.dims();

    x.reshape([a, b, 1, c, d])
        .expand([a, b, n_rep, c, d])
        .reshape([a, b * n_rep, c, d])
}

impl<B: Backend> Attention<B> {
    pub fn new(device: &B::Device, cfg: &Config, rotary_emb: RotaryEmbedding<B>) -> Self {
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
            rotary_emb: Some(rotary_emb),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        query.matmul(key.transpose())
            .div_scalar((self.head_dim as f32).sqrt())
    }

    pub fn forward(&self, x: Tensor<B, 3>, kvcache: &mut KVCache<B>) -> Tensor<B, 3> {
        let shape = x.dims();
        let b_sz = shape[0];
        let q_len = shape[1];

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

        let q = self.rotary_emb.as_ref().unwrap().forward(q, kvcache.offset());
        let k = self.rotary_emb.as_ref().unwrap().forward(k, kvcache.offset());

        kvcache.put(k, v);

        let k = kvcache.keys.clone().unwrap();
        let v = kvcache.values.clone().unwrap();

        let k = repeat_interleave(k, self.num_heads / self.num_kv_heads);
        let mut scroes = self.attn_scores(q, k);

        if q_len > 1 {
            let mask_attn = generate_autoregressive_mask(b_sz, q_len, &scroes.device());
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();

            scroes = scroes.mask_fill(
                mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2]).repeat_dim(1, self.num_heads),
                -1.0e4,
            );
        }
        let t = activation::softmax(scroes, 3);

        let v = repeat_interleave(v, self.num_heads / self.num_kv_heads);
        let o = t.matmul(v)
            .swap_dims(1, 2)
            .reshape([b_sz, q_len, self.num_heads * self.head_dim]);

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
    pub fn new(device: &B::Device, cfg: &Config, rotary_emb: RotaryEmbedding<B>) -> Self {
        Layer {
            attention: Attention::new(device, cfg, rotary_emb),
            mlp: Mlp::new(device, cfg),
            input_layernorm: RmsNormConfig::new(cfg.hidden_size).with_epsilon(cfg.rms_norm_eps).init(device),
            post_attention_layernorm: RmsNormConfig::new(cfg.hidden_size).with_epsilon(cfg.rms_norm_eps).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, kvcache: &mut KVCache<B>) -> Tensor<B, 3> {
        let residual = x.clone();

        let x = self.input_layernorm.forward(x);
        let x = self.attention.forward(x, kvcache);

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
    lm_head: Option<Linear<B>>
}

impl<B: Backend> Qwen3<B> {
    pub fn new(
        config: &Config,
        device: &B::Device,
    ) -> Self {
        let head_dim = config.head_dim();

        let embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);

        let rotary_emb = RotaryEmbedding::new(config.rope_theta, head_dim, config.max_position_embeddings, device);

        let layers = (0..config.num_hidden_layers)
            .map(|_| Layer::new(device, config, rotary_emb.clone()))
            .collect();

        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Qwen3 { embedding, layers, norm, lm_head: Some(lm_head) }
    }

    pub fn forward(&mut self, x: Tensor<B, 2, Int>, kvcaches: &mut [KVCache<B>]) -> Tensor<B, 3> {
        if self.lm_head.is_none() {
            let weight = self.embedding.weight.deref().clone();
            let weight = weight.transpose();

            self.lm_head = Some(Linear {
                weight: Param::initialized(ParamId::new(), weight),
                bias: None,
            });
        }

        let mut x = self.embedding.forward(x);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, &mut kvcaches[i]);
        }

        let x = self.norm.forward(x);
        let logits = self.lm_head.as_ref().unwrap().forward(x);

        activation::softmax(logits, 2)
    }
}