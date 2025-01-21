<a name="top"></a>
<h1 align="center">
  diffusion-rs
</h1>

<h3 align="center">
Blazingly fast inference of diffusion models.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/diffusion-rs/diffusion_rs_core/"><b>Latest Rust Documentation</b></a> | <a href="https://ericlbuehler.github.io/diffusion-rs/pyo3/diffusion_rs.html"><b>Latest Python Documentation</b></a> | <a href="https://discord.gg/DRcvs6z5vu"><b>Discord</b></a> |<a
href="README.md"><b>Chinese Doc</b></a> |
</p>


## Features
- Quantization
  - `bitsandbytes` format (fp4, nf4, and int8)
  - `GGUF` (2-8 bit quantization)
- Easy: Strong support for running [🤗 DDUF](https://huggingface.co/DDUF) models.
- Strong Apple Silicon support: support for the Metal, Accelerate, and ARM NEON frameworks
- Support for NVIDIA GPUs with CUDA
- AVX support for x86 CPUs
- Allow acceleration of models larger than the total VRAM size with offloading

Please do not hesitate to contact us with feature requests via [Github issues](https://github.com/EricLBuehler/diffusion-rs/issues)!

## Upcoming features
- 🚧 LoRA support
- 🚧 CPU + GPU inference with automatic offloading to allow partial acceleration of models larger than the total VRAM

## Installation
Check out the [installation guide](INSTALL.md) for details about installation.

## Examples
After [installing](#installation), you can try out these examples!

> Download the DDUF file here: `wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

**CLI:**
```bash
diffusion_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

More CLI examples [here](diffusion_rs_cli/README.md).

**Python:**

More Python examples [here](diffusion_rs_py/examples).

```py
from diffusion_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-dev-Q4-bnb.dduf"))

image_bytes = pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=50, guidance_scale=3.5
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
```

**Rust crate:**

Examples with the Rust crate: [here](diffusion_rs_examples/examples).

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
    vec!["Draw a picture of a sunrise.".to_string()],
    DiffusionGenerationParams {
        height: 720,
        width: 1280,
        num_steps: 50,
        guidance_scale: 3.5,
    },
)?;

let end = Instant::now();
println!("Took: {:.2}s", end.duration_since(start).as_secs_f32());

images[0].save("image.png")?;
```

## Support matrix
| Model | Supports DDUF | Supports quantized DDUF |
| -- | -- | -- |
| FLUX.1 Dev/Schnell | ✅ | ✅ |

## Contributing

- Anyone is welcome to contribute by opening PRs
  - See [good first issues](https://github.com/EricLBuehler/diffusion-rs/labels/good%20first%20issue) for a starting point!
- Collaborators will be invited based on past contributions
