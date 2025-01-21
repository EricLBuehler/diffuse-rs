
<a name="top"></a>
<h1 align="center">
  diffusion-rs
</h1>

<h3 align="center">
快速推理扩散模型。
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/diffusion-rs/diffusion_rs_core/"><b>最新 Rust 文档</b></a> | <a href="https://ericlbuehler.github.io/diffusion-rs/pyo3/diffusion_rs.html"><b>最新 Python 文档</b></a> | <a href="https://discord.gg/DRcvs6z5vu"><b>Discord</b></a> | <a
href="README.md"><b>英文文档</b></a> |
</p>

## 特性
- **量化**  
  - `bitsandbytes` 格式（fp4、nf4 和 int8）
  - `GGUF`（2-8 位量化）
- **简单易用**：强力支持运行 [🤗 DDUF](https://huggingface.co/DDUF) 模型。
- **强大的 Apple Silicon 支持**：支持 Metal、Accelerate 和 ARM NEON 框架。
- **支持 NVIDIA GPU 和 CUDA**
- **支持 x86 CPU 的 AVX**
- **支持通过卸载加速超出 VRAM 总容量的模型**

如有任何功能需求，欢迎通过 [Github issues](https://github.com/EricLBuehler/diffusion-rs/issues) 联系我们！

## 即将推出的功能
- 🚧 LoRA 支持
- 🚧 CPU + GPU 推理，自动卸载以允许部分加速超出总 VRAM 容量的模型

## 安装
查看 [安装指南](INSTALL.md)，了解详细的安装信息。

## 示例
在 [安装](#installation) 后，您可以尝试以下示例！

> 下载 DDUF 文件：`wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

**CLI:**
```bash
diffusion_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

更多 CLI 示例 [请见此处](diffusion_rs_cli/README.md)。

**Python:**

更多 Python 示例 [请见此处](diffusion_rs_py/examples)。

```py
from diffusion_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-dev-Q4-bnb.dduf"))

image_bytes = pipeline.forward(
    prompts=["画一幅日出景象的画。"],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=50, guidance_scale=3.5
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
```

**Rust crate:**

Rust crate 示例：[请见此处](diffusion_rs_examples/examples)。

```rust
use std::time::Instant;

use diffusion_rs_core::{DiffusionGenerationParams, ModelSource, ModelDType, Offloading, Pipeline, TokenSource};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

let filter = EnvFilter::builder()
    .with_default_directive(LevelFilter::INFO.into())
    .from_env_lossy();
tracing_subscriber::fmt().with_env_filter(filter).init();

let pipeline = Pipeline::load(
    ModelSource::dduf("FLUX.1-dev-Q4-bnb.dduf")?,
    false,
    TokenSource::CacheToken,
    None,
    None,
    &ModelDType::Auto,
)?;

let start = Instant::now();

let images = pipeline.forward(
    vec!["画一幅日出景象的画。".to_string()],
    DiffusionGenerationParams {
        height: 720,
        width: 1280,
        num_steps: 50,
        guidance_scale: 3.5,
    },
)?;

let end = Instant::now();
println!("耗时: {:.2}s", end.duration_since(start).as_secs_f32());

images[0].save("image.png")?;
```

## 支持矩阵
| 模型 | 支持 DDUF | 支持量化 DDUF |
| -- | -- | -- |
| FLUX.1 Dev/Schnell | ✅ | ✅ |

## 贡献

- 欢迎任何人通过提交 PR 来贡献代码
  - 请参阅 [good first issues](https://github.com/EricLBuehler/diffusion-rs/labels/good%20first%20issue) 开始！
- 基于过去的贡献，将邀请合作者加入项目。
