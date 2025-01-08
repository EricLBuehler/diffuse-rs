#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// T5 Text Model
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

use diffusion_rs_backend::{linear_no_bias, QuantMethod, QuantizedConfig};
use diffusion_rs_common::core::{DType, Device, Module, Result, Tensor, D};
use diffusion_rs_common::nn::{Activation, Embedding};
use diffusion_rs_common::{embedding, VarBuilder};
use float8::F8E4M3;
use serde::Deserialize;
use std::sync::Arc;

use super::{QuantizedModel, QuantizedModelLayer};

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ActivationWithOptionalGating {
    gated: bool,
    activation: diffusion_rs_common::nn::Activation,
}

fn deserialize_feed_forward_proj_activation<'de, D>(
    deserializer: D,
) -> std::result::Result<ActivationWithOptionalGating, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    match String::deserialize(deserializer)?.as_str() {
        "gated-gelu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: diffusion_rs_common::nn::Activation::NewGelu,
        }),
        "gated-silu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: diffusion_rs_common::nn::Activation::Silu,
        }),
        buf => {
            let activation = serde_plain::from_str(buf).map_err(serde::de::Error::custom)?;
            Ok(ActivationWithOptionalGating {
                gated: false,
                activation,
            })
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseActDense {
    wi: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    act: Activation,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let wi = linear_no_bias(cfg.d_model, cfg.d_ff, &cfg.quantization_config, vb.pp("wi"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, &cfg.quantization_config, vb.pp("wo"))?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
        })
    }
}

impl Module for T5DenseActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.wi.forward_autocast(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward_autocast(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: Arc<dyn QuantMethod>,
    wi_1: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    act: Activation,
}

impl T5DenseGatedActDense {
    fn load(vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let wi_0 = linear_no_bias(
            cfg.d_model,
            cfg.d_ff,
            &cfg.quantization_config,
            vb.pp("wi_0"),
        )?;
        let wi_1 = linear_no_bias(
            cfg.d_model,
            cfg.d_ff,
            &cfg.quantization_config,
            vb.pp("wi_1"),
        )?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, &cfg.quantization_config, vb.pp("wo"))?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: cfg.feed_forward_proj.activation,
        })
    }
}

impl Module for T5DenseGatedActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden_gelu = self.act.forward(&self.wi_0.forward_autocast(xs)?)?;
        let hidden_linear = self.wi_1.forward_autocast(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        let xs = self.wo.forward_autocast(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj.gated {
            (
                None,
                Some(T5DenseGatedActDense::load(vb.pp("DenseReluDense"), cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
        })
    }
}

impl Module for T5LayerFF {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5Attention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    o: Arc<dyn QuantMethod>,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
    use_cache: bool,
}

impl T5Attention {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &T5Config,
    ) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear_no_bias(cfg.d_model, inner_dim, &cfg.quantization_config, vb.pp("q"))?;
        let k = linear_no_bias(cfg.d_model, inner_dim, &cfg.quantization_config, vb.pp("k"))?;
        let v = linear_no_bias(cfg.d_model, inner_dim, &cfg.quantization_config, vb.pp("v"))?;
        let o = linear_no_bias(inner_dim, cfg.d_model, &cfg.quantization_config, vb.pp("o"))?;
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = embedding(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
            use_cache: cfg.use_cache && decoder,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Performs Self-attention (if key_value_states is None) or attention
        // over source sentence (provided by key_value_states).
        let kv_input = match key_value_states {
            None => xs,
            Some(key_value_states) => key_value_states,
        };
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = kv_input.dim(1)?;
        let q = self.q.forward_autocast(xs)?;
        let k = self.k.forward_autocast(kv_input)?;
        let v = self.v.forward_autocast(kv_input)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // TODO: Use flash_attn.
        let scores = { q.matmul(&k.t()?)? };
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    // This only handles the bidirectional case.
                    let kv_len = k.dim(2)?;
                    let (q_start, q_end) = match self.use_cache {
                        true => ((kv_len - q_len) as u32, kv_len as u32),
                        false => (0_u32, kv_len as u32),
                    };
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                    // TODO: position_bias_masked?
                }
            },
        };

        let attn_weights = { diffusion_rs_common::nn::ops::softmax_last_dim(&scores)? };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward_autocast(&attn_output)?;
        Ok((attn_output, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    fn load(h: bool, d: bool, vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, d, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            layer_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, mask)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerCrossAttention {
    fn load(decoder: bool, vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let cross_attention = T5Attention::load(false, decoder, vb.pp("EncDecAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            cross_attention,
            layer_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_hidden_states = self.layer_norm.forward(hidden_states)?;
        let (ys, position_bias) = self.cross_attention.forward(
            &normed_hidden_states,
            position_bias,
            Some(key_value_states),
            None,
        )?;
        let ys = (hidden_states + ys)?;
        Ok((ys, position_bias))
    }
}

trait TensorInfExtend {
    fn is_inf(&self) -> Result<Self>
    where
        Self: Sized;
    fn any(&self) -> Result<bool>;
}

impl TensorInfExtend for Tensor {
    fn is_inf(&self) -> Result<Self> {
        self.broadcast_eq(&Tensor::new(f64::INFINITY, self.device())?.to_dtype(self.dtype())?)
    }

    fn any(&self) -> Result<bool> {
        let sum = self.sum_all()?;
        match self.dtype() {
            DType::I8 => Ok(sum.to_scalar::<u8>()? == 0),
            DType::U8 => Ok(sum.to_scalar::<u8>()? == 0),
            DType::U32 => Ok(sum.to_scalar::<u32>()? == 0),
            DType::I16 => Ok(sum.to_scalar::<i16>()? == 0),
            DType::I32 => Ok(sum.to_scalar::<i32>()? == 0),
            DType::I64 => Ok(sum.to_scalar::<i64>()? == 0),
            DType::F16 => Ok(sum.to_scalar::<half::f16>()? == half::f16::from_f32_const(0.)),
            DType::BF16 => Ok(sum.to_scalar::<half::bf16>()? == half::bf16::from_f32_const(0.)),
            DType::F32 => Ok(sum.to_scalar::<f32>()? == 0.),
            DType::F64 => Ok(sum.to_scalar::<f64>()? == 0.),
            DType::F8E4M3 => Ok(sum.to_scalar::<F8E4M3>()? == F8E4M3::ZERO),
        }
    }
}

fn clamp_for_f16(xs: &Tensor) -> Result<Tensor> {
    let mut max = match xs.dtype() {
        DType::F16 => half::f16::MAX.to_f64_const() - 1000.,
        DType::BF16 => half::bf16::MAX.to_f64_const() - 1000.,
        DType::F32 => f32::MAX as f64 - 1000.,
        DType::F64 => f64::MAX - 1000.,
        _ => unreachable!(),
    };
    if xs.is_inf()?.any()? {
        max -= 1000.;
    }
    xs.clamp(-max, max)
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &T5Config,
    ) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn =
            T5LayerSelfAttention::load(has_relative_attention_bias, decoder, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(decoder, vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // TODO: Cache masks
        let mask = match self.cross_attn.is_some() {
            true => {
                let mask_len = xs.dim(1)?;
                // If the input seq length is 1, no need for a mask, this is also helpful to avoid shape
                // issues when using the KV cache in the decoder.
                if mask_len <= 1 {
                    None
                } else {
                    Some(get_mask(mask_len, xs.device())?)
                }
            }
            false => None,
        };
        let (mut xs, position_bias) = self.self_attn.forward(xs, position_bias, mask.as_ref())?;
        // Clamp for f16
        if xs.dtype() == DType::F16 {
            xs = clamp_for_f16(&xs)?;
        }
        if let Some(cross_attn) = &self.cross_attn {
            (xs, _) = cross_attn.forward(&xs, None, encoder_hidden_states.unwrap())?;
            // Clamp for f16
            if xs.dtype() == DType::F16 {
                xs = clamp_for_f16(&xs)?;
            }
        }
        let mut xs = self.ff.forward(&xs)?;
        // Clamp for f16
        if xs.dtype() == DType::F16 {
            xs = clamp_for_f16(&xs)?;
        }
        Ok((xs, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
}

impl T5Stack {
    fn load(
        decoder: bool,
        vb: VarBuilder,
        shared: &Arc<Embedding>,
        cfg: &T5Config,
    ) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, decoder, vb.pp(format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
                encoder_hidden_states,
            )?;
        }
        self.final_layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
}

impl T5EncoderModel {
    pub fn new(vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let shared_vb = if vb.contains_tensor("shared.weight") {
            vb.pp("shared")
        } else if vb.contains_tensor("decoder.embed_tokens") {
            vb.pp("decoder").pp("embed_tokens")
        } else {
            vb.pp("encoder").pp("embed_tokens")
        };
        let shared = embedding(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, cfg)?;
        Ok(Self { encoder })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input_ids, None)
    }
}

impl QuantizedModel for T5EncoderModel {
    fn match_devices_all_layers(&mut self, dev: &Device) -> Result<()> {
        self.encoder.shared = Arc::new(Embedding::new(
            self.encoder.shared.embeddings().to_device(dev)?,
            self.encoder.shared.hidden_size(),
        ));
        for block in &mut self.encoder.block {
            block.ff.layer_norm = T5LayerNorm {
                weight: block.ff.layer_norm.weight.to_device(dev)?,
                variance_epsilon: block.ff.layer_norm.variance_epsilon,
            };

            // Attention
            block.self_attn.layer_norm = T5LayerNorm {
                weight: block.self_attn.layer_norm.weight.to_device(dev)?,
                variance_epsilon: block.self_attn.layer_norm.variance_epsilon,
            };
            if let Some(emb) = &mut block.self_attn.self_attention.relative_attention_bias {
                *emb = Embedding::new(emb.embeddings().to_device(dev)?, emb.hidden_size());
            }
            if let Some(layer) = &mut block.cross_attn {
                layer.layer_norm = T5LayerNorm {
                    weight: layer.layer_norm.weight.to_device(dev)?,
                    variance_epsilon: layer.layer_norm.variance_epsilon,
                };
                if let Some(emb) = &mut layer.cross_attention.relative_attention_bias {
                    *emb = Embedding::new(emb.embeddings().to_device(dev)?, emb.hidden_size());
                }
            }
        }
        self.encoder.final_layer_norm = T5LayerNorm {
            weight: self.encoder.final_layer_norm.weight.to_device(dev)?,
            variance_epsilon: self.encoder.final_layer_norm.variance_epsilon,
        };
        Ok(())
    }

    fn aggregate_layers(&mut self) -> Result<Vec<QuantizedModelLayer>> {
        let mut layers = Vec::new();
        for block in &mut self.encoder.block {
            let mut layer_ct = Vec::new();
            // FF
            if let Some(layer) = &mut block.ff.dense_act {
                layer_ct.push(&mut layer.wi);
                layer_ct.push(&mut layer.wo);
            }
            if let Some(layer) = &mut block.ff.gated_dense_act {
                layer_ct.push(&mut layer.wi_1);
                layer_ct.push(&mut layer.wi_0);
                layer_ct.push(&mut layer.wo);
            }

            // Attention
            layer_ct.push(&mut block.self_attn.self_attention.q);
            layer_ct.push(&mut block.self_attn.self_attention.k);
            layer_ct.push(&mut block.self_attn.self_attention.v);
            layer_ct.push(&mut block.self_attn.self_attention.o);
            if let Some(layer) = &mut block.cross_attn {
                layer_ct.push(&mut layer.cross_attention.q);
                layer_ct.push(&mut layer.cross_attention.k);
                layer_ct.push(&mut layer.cross_attention.v);
                layer_ct.push(&mut layer.cross_attention.o);
            }

            layers.push(QuantizedModelLayer(layer_ct));
        }
        Ok(layers)
    }
}
