#![allow(unused)]
use super::GgmlDType;
use crate::core::{Error, MetalDevice, MetalStorage, Result};

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
}

impl QMetalStorage {
    pub fn zeros(_: &MetalDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize(&mut self, _src: &MetalStorage) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &MetalStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::core::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::core::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::core::Shape,
        _storage: &MetalStorage,
        _layout: &crate::core::Layout,
    ) -> Result<(MetalStorage, crate::core::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithMetalSupport)
}
