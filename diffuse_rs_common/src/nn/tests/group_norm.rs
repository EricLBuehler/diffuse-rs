/* Equivalent PyTorch code.
import torch
from torch.nn.functional import group_norm
t = torch.tensor(
        [[[-0.3034,  0.2726, -0.9659],
          [-1.1845, -1.3236,  0.0172],
          [ 1.9507,  1.2554, -0.8625],
          [ 1.0682,  0.3604,  0.3985],
          [-0.4957, -0.4461, -0.9721],
          [ 1.5157, -0.1546, -0.5596]],

         [[-1.6698, -0.4040, -0.7927],
          [ 0.3736, -0.0975, -0.1351],
          [-0.9461,  0.5461, -0.6334],
          [-1.0919, -0.1158,  0.1213],
          [-0.9535,  0.1281,  0.4372],
          [-0.2845,  0.3488,  0.5641]]])
print(group_norm(t, num_groups=2))
print(group_norm(t, num_groups=3))
*/
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use crate::core::test_utils::to_vec3_round;
use crate::core::{Device, Tensor};
use diffuse_rs_common::nn::{GroupNorm, Module};

#[test]
fn group_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::from_vec(vec![1f32; 6], 6, device)?;
    let b = Tensor::from_vec(vec![0f32; 6], 6, device)?;
    let gn2 = GroupNorm::new(w.clone(), b.clone(), 6, 2, 1e-5)?;
    let gn3 = GroupNorm::new(w, b, 6, 3, 1e-5)?;

    let input = Tensor::new(
        &[
            [
                [-0.3034f32, 0.2726, -0.9659],
                [-1.1845, -1.3236, 0.0172],
                [1.9507, 1.2554, -0.8625],
                [1.0682, 0.3604, 0.3985],
                [-0.4957, -0.4461, -0.9721],
                [1.5157, -0.1546, -0.5596],
            ],
            [
                [-1.6698, -0.4040, -0.7927],
                [0.3736, -0.0975, -0.1351],
                [-0.9461, 0.5461, -0.6334],
                [-1.0919, -0.1158, 0.1213],
                [-0.9535, 0.1281, 0.4372],
                [-0.2845, 0.3488, 0.5641],
            ],
        ],
        device,
    )?;
    assert_eq!(
        to_vec3_round(&gn2.forward(&input)?, 4)?,
        &[
            [
                [-0.1653, 0.3748, -0.7866],
                [-0.9916, -1.1220, 0.1353],
                [1.9485, 1.2965, -0.6896],
                [1.2769, 0.3628, 0.4120],
                [-0.7427, -0.6786, -1.3578],
                [1.8547, -0.3022, -0.8252]
            ],
            [
                [-1.9342, 0.0211, -0.5793],
                [1.2223, 0.4945, 0.4365],
                [-0.8163, 1.4887, -0.3333],
                [-1.7960, -0.0392, 0.3875],
                [-1.5469, 0.3998, 0.9561],
                [-0.3428, 0.7970, 1.1845]
            ]
        ]
    );
    assert_eq!(
        to_vec3_round(&gn3.forward(&input)?, 4)?,
        &[
            [
                [0.4560, 1.4014, -0.6313],
                [-0.9901, -1.2184, 0.9822],
                [1.4254, 0.6360, -1.7682],
                [0.4235, -0.3800, -0.3367],
                [-0.3890, -0.3268, -0.9862],
                [2.1325, 0.0386, -0.4691]
            ],
            [
                [-1.8797, 0.0777, -0.5234],
                [1.2802, 0.5517, 0.4935],
                [-1.0102, 1.5327, -0.4773],
                [-1.2587, 0.4047, 0.8088],
                [-1.9074, 0.1691, 0.7625],
                [-0.6230, 0.5928, 1.0061]
            ]
        ]
    );

    Ok(())
}
