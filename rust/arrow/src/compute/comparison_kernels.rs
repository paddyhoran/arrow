// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines basic arithmetic kernels for `PrimitiveArrays`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the documentation
//! [here] (https://doc.rust-lang.org/stable/std/arch/) for more information.

use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use crate::array::*;
use crate::array_data::ArrayData;
use crate::compute::util::apply_bin_op_to_option_bitmap;
use crate::datatypes::{DataType, BooleanType, ArrowNumericType};
use crate::error::{ArrowError, Result};
use crate::builder::*;

/// SIMD vectorized version of `math_op` above.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_comp_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<BooleanArray>
    where
        T: ArrowNumericType,
        F: Fn(T::Simd, T::Simd) -> T::SimdMask,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }

    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left.data().null_bitmap(),
        right.data().null_bitmap(),
        |a, b| a & b,
    )?;

    let lanes = T::lanes();
    let mut result = BooleanBufferBuilder::new(left.len());

    for i in (0..left.len()).step_by(lanes) {
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_right = T::load(right.value_slice(i, lanes));
        let simd_result = op(simd_left, simd_right);
        for i in 0..lanes {
            result.append(T::mask_extract(&simd_result, i)).unwrap();
        }
    }

    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![result.finish()],
        vec![],
    );
    Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
}

/// Perform `left == right` operation on two arrays.
pub fn eq<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
    where
        T: ArrowNumericType,
{
    simd_comp_op(left, right, |a, b| T::eq(a, b))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Int32Array;

    #[test]
    fn test_primitive_array_eq() {
        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
        let c = eq(&a, &b).unwrap();
        assert_eq!(false, c.value(0));
        assert_eq!(false, c.value(1));
        assert_eq!(true, c.value(2));
        assert_eq!(false, c.value(3));
        assert_eq!(false, c.value(4));
    }

//    #[test]
//    fn test_primitive_array_neq() {
//        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
//        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
//        let c = neq(&a, &b).unwrap();
//        assert_eq!(true, c.value(0));
//        assert_eq!(true, c.value(1));
//        assert_eq!(false, c.value(2));
//        assert_eq!(true, c.value(3));
//        assert_eq!(true, c.value(4));
//    }
//
//    #[test]
//    fn test_primitive_array_lt() {
//        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
//        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
//        let c = lt(&a, &b).unwrap();
//        assert_eq!(false, c.value(0));
//        assert_eq!(false, c.value(1));
//        assert_eq!(false, c.value(2));
//        assert_eq!(true, c.value(3));
//        assert_eq!(true, c.value(4));
//    }
//
//    #[test]
//    fn test_primitive_array_lt_nulls() {
//        let a = Int32Array::from(vec![None, None, Some(1)]);
//        let b = Int32Array::from(vec![None, Some(1), None]);
//        let c = lt(&a, &b).unwrap();
//        assert_eq!(false, c.value(0));
//        assert_eq!(true, c.value(1));
//        assert_eq!(false, c.value(2));
//    }
//
//    #[test]
//    fn test_primitive_array_lt_eq() {
//        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
//        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
//        let c = lt_eq(&a, &b).unwrap();
//        assert_eq!(false, c.value(0));
//        assert_eq!(false, c.value(1));
//        assert_eq!(true, c.value(2));
//        assert_eq!(true, c.value(3));
//        assert_eq!(true, c.value(4));
//    }
//
//    #[test]
//    fn test_primitive_array_lt_eq_nulls() {
//        let a = Int32Array::from(vec![None, None, Some(1)]);
//        let b = Int32Array::from(vec![None, Some(1), None]);
//        let c = lt_eq(&a, &b).unwrap();
//        assert_eq!(true, c.value(0));
//        assert_eq!(true, c.value(1));
//        assert_eq!(false, c.value(2));
//    }
//
//    #[test]
//    fn test_primitive_array_gt() {
//        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
//        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
//        let c = gt(&a, &b).unwrap();
//        assert_eq!(true, c.value(0));
//        assert_eq!(true, c.value(1));
//        assert_eq!(false, c.value(2));
//        assert_eq!(false, c.value(3));
//        assert_eq!(false, c.value(4));
//    }
//
//    #[test]
//    fn test_primitive_array_gt_nulls() {
//        let a = Int32Array::from(vec![None, None, Some(1)]);
//        let b = Int32Array::from(vec![None, Some(1), None]);
//        let c = gt(&a, &b).unwrap();
//        assert_eq!(false, c.value(0));
//        assert_eq!(false, c.value(1));
//        assert_eq!(true, c.value(2));
//    }
//
//    #[test]
//    fn test_primitive_array_gt_eq() {
//        let a = Int32Array::from(vec![8, 8, 8, 8, 8]);
//        let b = Int32Array::from(vec![6, 7, 8, 9, 10]);
//        let c = gt_eq(&a, &b).unwrap();
//        assert_eq!(true, c.value(0));
//        assert_eq!(true, c.value(1));
//        assert_eq!(true, c.value(2));
//        assert_eq!(false, c.value(3));
//        assert_eq!(false, c.value(4));
//    }
//
//    #[test]
//    fn test_primitive_array_gt_eq_nulls() {
//        let a = Int32Array::from(vec![None, None, Some(1)]);
//        let b = Int32Array::from(vec![None, Some(1), None]);
//        let c = gt_eq(&a, &b).unwrap();
//        assert_eq!(true, c.value(0));
//        assert_eq!(false, c.value(1));
//        assert_eq!(true, c.value(2));
//    }

}
